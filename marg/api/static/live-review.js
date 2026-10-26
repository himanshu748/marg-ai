"use strict";
(() => {
  const el = (id) => document.getElementById(id);
  let current = null, timer = null;
  async function request(path, options = {}) {
    const response = await fetch(path, { credentials: "same-origin", ...options });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Live review unavailable.");
    return data;
  }
  function status(text) { el("liveStatus").textContent = text; }
  function render(job) {
    const result = job.result;
    if (!result) return;
    const passed = result.audit?.passed === true && job.status === "succeeded";
    const provider = result.provider || {};
    const checks = result.audit?.checks || result.audit?.rules || [];
    const trace = (result.trace || []).map((entry, index) => `<li><span class="timeline-number">${index + 1}</span><div class="timeline-content"><strong>${esc(entry.tool)}</strong><p>${esc(entry.model_text || entry.output?.text || "")}</p>${entry.output?.crop_path ? `<img class="timeline-crop" alt="Road region re-inspected during this live review" src="/api/live-reviews/${esc(job.job_id)}/media/agent_crops/${esc(String(entry.output.crop_path).split("/").pop())}">` : ""}<details><summary>Tool inputs and results</summary><pre>${esc(JSON.stringify({input:entry.input,output:entry.output},null,2))}</pre></details></div></li>`).join("");
    const orders = (result.work_orders || []).map((order, index) => {
      const id = order.work_order_id, decision = job.decisions?.[id];
      return `<article class="order"><div><h4>${esc(order.title || order.summary || "Repair recommendation")}</h4><p>${esc(order.reason || "")}</p><p>${decision ? `${esc(decision.decision === "approve" ? "Approved" : "Rejected")} in your private review. No repair dispatched.` : "Awaiting your decision."}</p>${decision ? `<p>${esc(decision.note)}</p>` : `<label for="liveNote${index}">Decision note</label><textarea id="liveNote${index}" data-live-note="${esc(id)}" maxlength="2000" rows="2"></textarea><div class="decision-actions"><button class="button primary" data-live-order="${esc(id)}" data-live-decision="approve" ${passed ? "" : "disabled"}>Approve</button><button class="button quiet" data-live-order="${esc(id)}" data-live-decision="reject" ${passed ? "" : "disabled"}>Reject</button></div>`}</div></article>`;
    }).join("");
    el("liveResult").innerHTML = `<h4>Live DeepSeek review · ${esc(titleFor(job.survey_id))}</h4><details><summary>Model summary</summary><p>${esc(result.summary || "Review result")}</p></details><p>${provider.live_inference ? "Live inference verified" : "No completed inference"} · ${number(result.tool_call_count)} tool calls · ${passed ? "Policy audit passed" : "Policy audit did not pass; decisions are blocked"}${checks.length ? ` · ${checks.filter(c => c.passed ?? c.ok).length}/${checks.length} checks` : ""}</p><p>These results belong to this live run. The recorded survey above is unchanged.</p><div class="live-orders">${orders}</div><details><summary>Live model and OpenCV activity</summary><ol class="timeline">${trace}</ol></details><button id="exportLiveReview" class="button quiet" type="button">Download live review JSON</button>`;
    el("exportLiveReview").addEventListener("click", () => downloadFile(`margai-live-${job.job_id}.json`, JSON.stringify(job,null,2), "application/json"));
    el("liveResult").querySelectorAll("[data-live-decision]").forEach(button => button.addEventListener("click", async () => {
      const order = button.dataset.liveOrder;
      const note = [...el("liveResult").querySelectorAll("[data-live-note]")].find(input => input.dataset.liveNote === order)?.value || "";
      el("liveResult").querySelectorAll("[data-live-decision]").forEach(b => b.disabled = true);
      try {
        current = await request(`/api/live-reviews/${job.job_id}/decisions/${encodeURIComponent(order)}`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision:button.dataset.liveDecision,note})});
        render(current); status("Decision saved in your private live review. No repair dispatched.");
      } catch (error) { render(current); status(error.message); }
    }));
  }
  async function poll() {
    if (!current) return;
    try {
      current = await request(`/api/live-reviews/${current.job_id}`);
      const done = ["succeeded", "failed"].includes(current.status);
      status(done ? current.status === "succeeded" ? "Live review complete. Inspect the policy audit and recommendations below." : current.error || "Live review unavailable. Recorded evidence is unchanged." : "Live DeepSeek review in progress. The model is choosing and checking OpenCV tools.");
      if (done) { el("runLiveReview").disabled = false; render(current); }
      else timer = setTimeout(poll, 1800);
    } catch (error) { status(error.message); el("runLiveReview").disabled = false; sessionStorage.removeItem("marg-live-review"); }
  }
  el("runLiveReview").addEventListener("click", async () => {
    el("runLiveReview").disabled = true; clearTimeout(timer);
    status("Starting your private live review…");
    try {
      current = await request("/api/live-reviews", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({survey_id:el("liveSurvey").value})});
      el("liveResult").replaceChildren();
      sessionStorage.setItem("marg-live-review", current.job_id); poll();
    } catch (error) { status(error.message); el("runLiveReview").disabled = false; }
  });
  request("/api/capabilities").then(caps => {
    const live = caps.live_reviews;
    if (!live) return;
    el("liveReviewSection").hidden = false;
    el("runLiveReview").disabled = !live.enabled;
    const survey = new URL(location.href).searchParams.get("survey");
    if (live.allowed_surveys.includes(survey)) el("liveSurvey").value = survey;
    if (!live.enabled) status("The live demo allowance is unavailable or expired. Recorded examples remain available.");
    const saved = sessionStorage.getItem("marg-live-review");
    if (saved) { current = {job_id:saved}; el("runLiveReview").disabled = true; poll(); }
  }).catch(() => {});
})();
