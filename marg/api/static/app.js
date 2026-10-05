"use strict";
const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ],
  );
const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const number = (value, digits = 0) =>
  Number.isFinite(Number(value))
    ? Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
    : "—";
const classNames = {
  D00: "Longitudinal crack",
  D10: "Transverse crack",
  D20: "Alligator crack",
  D40: "Pothole",
};
const titleFor = (value) =>
  String(value || "Untitled survey")
    .replace(/[_-]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
const idLabel = (id) => `#${String(Number(id) + 1).padStart(3, "0")}`;
const state = {
  caps: null,
  surveys: [],
  id: null,
  result: null,
  agent: null,
  trace: [],
  selected: null,
  view: "inspection",
  uncertain: false,
  sequence: 0,
  job: null,
  jobTimer: null,
  jobFailures: 0,
  busy: false,
  map: null,
  markers: null,
  markerById: new Map(),
  decision: null,
  authenticated: false,
};
let toastTimer;
class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}
async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: {
      ...(options.body instanceof FormData
        ? {}
        : { Accept: "application/json" }),
      ...options.headers,
    },
  });
  let data;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    const detail = data?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message ||
          (Array.isArray(detail)
            ? detail.map((d) => d.msg).join("; ")
            : null) ||
          `Request failed (${response.status}). Please try again.`;
    if (response.status === 401) {
      state.authenticated = false;
      applyCapabilities();
    }
    throw new ApiError(message, response.status, detail);
  }
  return data;
}
function notify(message, error = false) {
  $("notice").hidden = false;
  $("notice").classList.toggle("error", error);
  $("notice").textContent = message;
}
function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => ($("toast").hidden = true), 5000);
}
function showDialog(id) {
  if (!$(id).open) $(id).showModal();
}
function empty(title, copy) {
  return `<div class="empty-state"><h4>${esc(title)}</h4><p>${esc(copy)}</p></div>`;
}
function media(category, name, id = state.id) {
  return `/api/surveys/${encodeURIComponent(id)}/${category}/${encodeURIComponent(String(name).split("/").pop())}`;
}
function severity(value) {
  return `<span class="severity ${value >= 4 ? "high" : value === 3 ? "medium" : ""}">${value >= 4 ? "High" : value === 3 ? "Moderate" : "Low"} · ${esc(value)}/5</span>`;
}
function isReadonly() {
  return (
    !state.caps ||
    state.caps.read_only ||
    (state.caps.authentication_required && !state.authenticated)
  );
}
function reviewState() {
  return MargReview.summary(state.result, state.agent);
}
function renderReviewBrief() {
  if (!state.result) return;
  const s = reviewState();
  $("nextStepTitle").textContent = state.busy ? "Review work is in progress" : s.next.title;
  $("nextStepDetail").textContent = state.busy ? "Keep exploring the saved evidence. Decisions unlock after the review completes and its policy checks pass." : s.next.detail;
  $("nextStepAction").textContent = s.next.label;
  $("nextStepAction").disabled = state.busy || (s.next.action === "review" && isReadonly());
  $("stepEvidence").className = "complete";
  $("stepReview").className = s.valid ? "complete" : "current";
  $("stepReview").textContent = s.valid ? "2 · Policy checks passed" : "2 · Checked review needed";
  $("stepDecision").className = s.pending ? "current" : s.valid ? "complete" : "";
  $("stepDecision").textContent = s.pending ? `3 · ${s.pending} awaiting decision` : s.valid ? `3 · ${s.decided} decisions recorded` : "3 · Human decision";
  $("surveyStatus").textContent = state.busy ? "Review in progress" : !s.valid ? "Review required" : s.pending ? "Decision required" : "Review checked";
}
function applyCapabilities() {
  const c = state.caps;
  if (!c) return;
  const locked = c.authentication_required && !state.authenticated;
  $("newSurvey").disabled = !c.uploads?.enabled || locked;
  $("emptyUpload").disabled = $("newSurvey").disabled;
  $("runReview").disabled = isReadonly() || !state.result || state.busy;
  $("modeBadge").innerHTML =
    `<span></span>${c.read_only ? "Read-only workspace" : c.agent?.bedrock_enabled ? "Bedrock enabled" : "Local review"}`;
  $("storageLabel").textContent =
    c.storage === "aws" ? "AWS workspace" : "Local workspace";
  $("accessButton").hidden = !c.authentication_required;
  $("accessButton").textContent = state.authenticated
    ? "Lock workspace"
    : "Unlock workspace";
  const chosenMode = $("reviewMode").value;
  $("reviewMode").innerHTML = (c.agent?.modes || ["mock"])
    .map(
      (mode) =>
        `<option value="${esc(mode)}">${mode === "bedrock" ? "Bedrock review" : "Deterministic review"}</option>`,
    )
    .join("");
  if ((c.agent?.modes || ["mock"]).includes(chosenMode))
    $("reviewMode").value = chosenMode;
  $("reviewMode").disabled = isReadonly() || state.busy;
  $("exportButton").disabled = !state.result || state.busy;
  document
    .querySelectorAll("[data-decision]")
    .forEach((button) => (button.disabled = isReadonly() || state.busy || !MargReview.validReview(state.agent)));
  $("reviewDescription").textContent = c.read_only
    ? "This workspace is read-only. Existing evidence and review records are available to inspect."
    : "Run a deterministic review to inspect uncertainty and prepare work orders.";
  $("videoHint").textContent =
    `MP4, MOV, AVI, MKV or WebM · up to ${number((c.uploads?.max_video_bytes || 268435456) / 1048576)} MB`;
  $("videoFile").accept = (c.uploads?.accepted_video_extensions || [".mp4", ".mov", ".avi", ".mkv", ".webm"]).join(",");
  renderReviewBrief();
}
function switchView(view) {
  state.view = view;
  const labels = {
    inspection: [
      "Survey review",
      "From road evidence to the right repair decision.",
    ],
    orders: [
      "Work orders",
      "Turn verified observations into considered repair decisions.",
    ],
    activity: [
      "Review activity",
      "Follow the evidence behind every recommendation.",
    ],
  };
  $("pageTitle").textContent = labels[view][0];
  $("breadcrumbView").textContent = labels[view][0];
  $("pageDescription").textContent = labels[view][1];
  document.title = `MargAI — ${labels[view][0]}`;
  for (const v of Object.keys(labels)) $(`${v}View`).hidden = v !== view;
  document.querySelectorAll("[data-view]").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  setSidebar(false);
  if (view === "inspection" && !$("mapStage").hidden)
    requestAnimationFrame(() => state.map?.invalidateSize());
}
function renderSurveyList() {
  $("surveyList").innerHTML = state.surveys.length
    ? state.surveys
        .map(
          (s) =>
            `<button class="survey-option ${s.id === state.id ? "active" : ""}" data-survey="${esc(s.id)}" ${s.id === state.id ? 'aria-current="true"' : ""}>${icon("road")}<span><strong>${esc(titleFor(s.name || s.id.replace(/-[a-f0-9]{10}$/, "")))}</strong><small>${["done", "succeeded"].includes(s.status) ? `${number(s.instances)} defects · ready` : esc(s.status || "Ready")}</small></span></button>`,
        )
        .join("")
    : '<p class="small muted">No surveys yet. Add your first video to begin.</p>';
}
async function loadSurveys(preferred = state.id) {
  state.surveys = await api("/api/surveys");
  renderSurveyList();
  if (!state.surveys.length) {
    state.id = null;
    state.result = null;
    $("surveyWorkspace").hidden = true;
    $("emptyWorkspace").hidden = false;
    applyCapabilities();
    return;
  }
  $("emptyWorkspace").hidden = true;
  const selected =
    state.surveys.find((s) => s.id === preferred) ||
    state.surveys.find((s) => ["done", "succeeded"].includes(s.status)) ||
    state.surveys[0];
  await selectSurvey(selected.id);
}
async function optional(path) {
  try {
    return await api(path);
  } catch (error) {
    if (error.status === 404) return null;
    throw error;
  }
}
async function selectSurvey(id) {
  const seq = ++state.sequence;
  if (id !== state.id) {
    $("searchDefects").value = "";
    $("outcomeFilter").value = "all";
    state.uncertain = false;
    $("uncertainFilter").setAttribute("aria-pressed", "false");
  }
  state.id = id;
  const selectedUrl = new URL(location.href);
  selectedUrl.searchParams.set("survey", id);
  history.replaceState({}, "", selectedUrl);
  state.result = null;
  state.agent = null;
  state.trace = [];
  state.selected = null;
  renderSurveyList();
  $("notice").hidden = true;
  $("surveyWorkspace").hidden = true;
  applyCapabilities();
  const listed = state.surveys.find((s) => s.id === id);
  if (listed && !["done", "succeeded"].includes(listed.status)) {
    const status = await api(`/api/surveys/${encodeURIComponent(id)}/status`);
    if (seq !== state.sequence) return;
    if (status.status === "failed")
      notify(
        status.error?.message ||
          status.error ||
          "Survey processing failed. Check the video format and upload again.",
        true,
      );
    else
      notify(
        `Survey ${status.status || "processing"}. You can inspect another survey while it finishes.`,
      );
    if (
      status.job_id &&
      !["failed", "succeeded", "done"].includes(status.status)
    )
      trackJob({ ...status, status_url: `/api/jobs/${status.job_id}` });
    return;
  }
  try {
    const base = `/api/surveys/${encodeURIComponent(id)}`;
    const [result, agent, trace, status] = await Promise.all([
      api(base),
      optional(base + "/agent"),
      optional(base + "/trace"),
      optional(base + "/status"),
    ]);
    if (seq !== state.sequence) return;
    state.result = result;
    state.agent = agent;
    state.trace = trace || [];
    state.selected =
      [...result.instances].sort(
        (a, b) => Number(MargReview.outcome(a, agent, result).code === "dismissed") - Number(MargReview.outcome(b, agent, result).code === "dismissed") || b.severity - a.severity || a.id - b.id,
      )[0]?.id ?? null;
    $("surveyWorkspace").hidden = false;
    $("surveyName").textContent = titleFor(
      listed?.name || id.replace(/-[a-f0-9]{10}$/, ""),
    );
    $("surveyMeta").textContent =
      `${number(result.keyframes)} keyframes · ${number(result.segments.length)} road segments · ${result.geo_source === "synthetic" ? "Synthetic locations" : titleFor(result.geo_source) + " locations"}`;
    $("surveyStatus").textContent =
      agent?.status === "unavailable"
        ? "Review unavailable"
        : agent?.status === "failed"
          ? "Review needs attention"
          : "Ready for review";
    updateMetrics();
    renderDefects();
    renderEvidence();
    renderFrames();
    renderOrders();
    renderTrace();
    applyCapabilities();
    $("provenanceLabel").textContent = providerLabel();
    if (agent?.error)
      notify(
        agent.error.message ||
          "The latest review could not complete. Evidence remains available.",
        true,
      );
    if (
      status?.agent_job &&
      ["queued", "running"].includes(status.agent_job.status) &&
      !state.job
    )
      trackJob({
        ...status.agent_job,
        status_url: `/api/jobs/${status.agent_job.job_id}`,
      });
    if (!$("mapStage").hidden) renderMap();
  } catch (error) {
    if (seq !== state.sequence) return;
    notify(`Could not load this survey. ${error.message}`, true);
  }
}
function providerLabel() {
  const provider = state.agent?.provider;
  if (provider?.kind === "mock")
    return "Deterministic review · no live model inference";
  if (provider?.kind === "bedrock")
    return `Bedrock · ${provider.model_id || "configured model"}${provider.live_inference ? "" : " · no completed inference"}`;
  return state.agent
    ? "Recorded review · provider not recorded"
    : "No review has been run";
}
function updateMetrics() {
  const r = state.result;
  if (!r) return;
  const s = reviewState(), pending = s.pending;
  $("metricDefects").textContent = number(r.instances.length);
  $("metricDetections").textContent =
    `From ${number(Object.values(r.detections).reduce((a, b) => a + Number(b), 0))} frame detections`;
  $("metricUncertain").textContent = number(s.unresolved);
  $("metricReviewDetail").textContent = s.valid ? `${s.dismissed} dismissed · ${s.resurveys} resurveys` : "Low confidence · review needed";
  $("metricOrders").textContent = number(pending);
  $("navOrderCount").textContent = number(pending);
  $("metricRuntime").textContent = `${number(r.metrics.runtime_s, 1)}s`;
  $("metricFrames").textContent = `${number(r.frames)} source frames`;
  renderReviewBrief();
}
function filteredDefects() {
  const query = $("searchDefects").value.toLowerCase().trim();
  const outcomes = reviewState().outcomes;
  return [...(state.result?.instances || [])]
    .filter(
      (i) =>
        (!state.uncertain || outcomes.get(i.id)?.needsReview) &&
        MargReview.matchesOutcome(i, state.agent, state.result, $("outcomeFilter").value) &&
        `${classNames[i.class_name] || i.class_name} ${i.class_name} ${idLabel(i.id)} ${i.id}`
          .toLowerCase()
          .includes(query),
    )
    .sort((a, b) =>
      $("sortDefects").value === "confidence"
        ? a.fused_conf - b.fused_conf
        : $("sortDefects").value === "id"
          ? a.id - b.id
          : b.severity - a.severity || a.id - b.id,
    );
}
function renderDefects() {
  const defects = filteredDefects();
  const outcomes = reviewState().outcomes;
  $("defectCount").textContent = number(defects.length);
  $("defectList").innerHTML = defects.length
    ? defects
        .map(
          (i) =>
            `<button class="defect-row ${outcomes.get(i.id).code} ${i.id === state.selected ? "active" : ""}" data-defect="${i.id}" aria-pressed="${i.id === state.selected}">${i.evidence_path ? `<img class="defect-thumb" src="${media("evidence", i.evidence_path)}" alt="" loading="lazy">` : `<span class="defect-thumb"></span>`}<span class="defect-info"><strong>${esc(classNames[i.class_name] || i.class_name)} ${idLabel(i.id)}</strong><small class="outcome-label ${outcomes.get(i.id).code}">${esc(outcomes.get(i.id).label)}</small></span><span class="defect-meta">${severity(i.severity)}<small>${number(i.fused_conf * 100)}% confidence</small></span></button>`,
        )
        .join("")
    : empty(
        state.result?.instances.length
          ? "No matching defects"
          : "No defects detected",
        state.result?.instances.length
          ? "Try another search or clear the review and outcome filters."
          : "This survey did not retain road-damage instances. You can still inspect its keyframes and review activity.",
      );
}
function selectDefect(id) {
  state.selected = id;
  renderDefects();
  renderEvidence();
  if (!$("mapStage").hidden) {
    const marker = state.markerById.get(id);
    marker?.openPopup();
  }
}
function setEvidence(src, title, caption) {
  $("evidenceTitle").textContent = title;
  $("evidenceImage").hidden = !src;
  $("imageEmpty").hidden = !!src;
  $("expandEvidence").hidden = !src;
  $("imageBadge").hidden = !src;
  if (src) {
    $("evidenceImage").src = src;
    $("evidenceImage").alt = title + " — recorded road evidence";
    $("imageEmpty").textContent = "Loading evidence…";
  }
  $("evidenceCaption").textContent = caption;
  $("fullImageCaption").textContent = caption;
}
function renderEvidence() {
  const r = state.result;
  if (!r) return;
  const i = r.instances.find((i) => i.id === state.selected);
  $("selectedDetails").classList.toggle("overview", !i);
  $("geoWarning").textContent =
    r.geo_source === "synthetic" ? "Synthetic GPS · illustrative location" : "";
  if (!i) {
    const first = r.keyframe_paths[0];
    setEvidence(
      first ? media("keyframes", first) : null,
      "Survey evidence",
      first ? "Keyframe 1 · survey overview" : "No evidence frames available",
    );
    $("imageBadge").textContent = "Survey keyframe";
    $("selectedDetails").innerHTML =
      '<p class="small muted">No defects were detected in this survey.</p>';
    return;
  }
  const title = `${classNames[i.class_name] || i.class_name} ${idLabel(i.id)}`;
  setEvidence(
    i.evidence_path ? media("evidence", i.evidence_path) : null,
    title,
    `Best observation · frame ${i.best_frame_idx >= 0 ? i.best_frame_idx : "not recorded"}`,
  );
  $("imageBadge").textContent = `${i.class_name} · severity ${i.severity}/5`;
  $("selectedDetails").innerHTML =
    `<div><span>Fused confidence</span><strong>${number(i.fused_conf * 100)}%</strong><small>${i.fused_conf < 0.55 ? "Close inspection required" : "Review alongside evidence"}</small></div><div><span>Estimated area</span><strong>${number(i.area_m2, 2)} m²</strong><small>Assumed road plane</small></div><div><span>Severity</span><strong>${number(i.severity)} <small>/ 5</small></strong><small>${i.severity >= 4 ? "High priority for review" : i.severity === 3 ? "Moderate damage" : "Lower severity"}</small></div>`;
  const o = MargReview.outcome(i, state.agent, r);
  $("selectedDetails").firstElementChild.querySelector("small").textContent = o.needsReview ? "Unresolved evidence" : "Detection confidence · before review";
  if (o.code === "dismissed") $("selectedDetails").children[2].lastElementChild.textContent = "Detection estimate · dismissed";
  $("selectedDetails").insertAdjacentHTML("beforeend", `<div class="evidence-outcome"><strong class="outcome-label ${o.code}">${esc(o.label)}</strong><p>${esc(o.detail)}</p></div>`);
  document
    .querySelectorAll(".frame-thumb")
    .forEach((b) => b.classList.remove("active"));
}
function renderFrames() {
  const frames = state.result?.keyframe_paths || [];
  $("frameCount").textContent = `${number(frames.length)} extracted`;
  $("frameStrip").innerHTML = frames
    .map(
      (path, index) =>
        `<button class="frame-thumb" data-frame="${index}" aria-label="View keyframe ${index + 1}"><img src="${media("keyframes", path)}" alt="" loading="lazy"><span>${String(index + 1).padStart(2, "0")}</span></button>`,
    )
    .join("");
}
function showFrame(index) {
  const path = state.result?.keyframe_paths[index];
  if (!path) return;
  $("showEvidence").click();
  setEvidence(
    media("keyframes", path),
    `Survey keyframe ${index + 1}`,
    `Keyframe ${index + 1} of ${state.result.keyframes} · full survey frame`,
  );
  $("imageBadge").textContent = "Survey keyframe";
  state.selected = null;
  renderDefects();
  $("selectedDetails").classList.add("overview");
  $("selectedDetails").innerHTML =
    '<p class="small muted">Full survey frame. Select a defect to see its confidence, estimated area, and severity.</p>';
  document
    .querySelectorAll(".frame-thumb")
    .forEach((b) =>
      b.classList.toggle("active", Number(b.dataset.frame) === index),
    );
}
function renderMap() {
  if (!state.result) return;
  if (!window.L) {
    $("mapError").hidden = false;
    return;
  }
  if (!state.map) {
    state.map = L.map("map", { zoomControl: true }).setView([20, 78], 5);
    state.markers = L.layerGroup().addTo(state.map);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    })
      .on("tileerror", () => ($("mapError").hidden = false))
      .addTo(state.map);
  }
  state.markers.clearLayers();
  state.markerById.clear();
  const points = [];
  for (const i of state.result.instances) {
    if (!Number.isFinite(i.lat) || !Number.isFinite(i.lon)) continue;
    points.push([i.lat, i.lon]);
    const marker = L.circleMarker([i.lat, i.lon], {
      radius: 9,
      fillColor: i.severity >= 4 ? "#e7977c" : "#6adbc6",
      color: "#14231f",
      weight: 2,
      fillOpacity: 1,
    });
    marker.bindPopup(
      `<strong>${esc(classNames[i.class_name] || i.class_name)} ${idLabel(i.id)}</strong><br>Severity ${i.severity}/5 · ${number(i.fused_conf * 100)}% confidence<br>${state.result.geo_source === "synthetic" ? "Synthetic location" : `${number(i.lat, 5)}, ${number(i.lon, 5)}`}`,
    );
    marker.on("click", () => selectDefect(i.id));
    marker.addTo(state.markers);
    state.markerById.set(i.id, marker);
  }
  requestAnimationFrame(() => {
    state.map.invalidateSize();
    if (points.length)
      state.map.fitBounds(points, { padding: [40, 40], maxZoom: 18 });
  });
}
function renderOrders() {
  const all = state.agent?.work_orders || [];
  const filter = $("orderFilter").value;
  const orders = all.filter((o) => filter === "all" || o.status === filter);
  if (!orders.length) {
    $("orderList").innerHTML = empty(
      all.length
        ? "No work orders in this view"
        : state.agent
          ? "No work orders required"
          : "No review yet",
      all.length
        ? "Choose another status to see the other work orders."
        : state.agent
          ? "This review did not produce work orders. Inspect activity for the reasoning."
          : "Run a review from the survey view to prepare evidence-backed work orders.",
    );
    return;
  }
  $("orderList").innerHTML = orders
    .map((o) => {
      const pending = o.status === "pending_approval",
        approved = o.status === "approved",
        rejected = o.status === "rejected";
      const status = approved
        ? "Approved"
        : rejected
          ? "Rejected"
          : pending
            ? "Awaiting approval"
            : o.status === "review_blocked"
              ? "Review blocked"
              : titleFor(o.status);
      return `<article class="order"><div><div class="order-meta"><span class="severity ${o.priority === "high" ? "high" : o.priority === "medium" ? "medium" : ""}">${esc(titleFor(o.priority))} priority</span><span>Segment ${esc(Number(o.segment_id) + 1)}</span><span>${esc(String(o.work_order_id).slice(0, 8))}</span></div><h4>${esc(o.title || o.summary || "Road repair work order")}</h4><p>${esc(o.reason || o.summary)}</p><div class="order-evidence">${(o.instances || []).map((i) => `<button data-order-evidence="${esc(i.evidence || "")}" data-evidence-title="${esc((classNames[i.class] || i.class) + " " + idLabel(i.id))}" ${!i.evidence ? "disabled" : ""}>${i.evidence ? `<img src="${media("evidence", i.evidence)}" alt="${esc(classNames[i.class] || i.class)} evidence" loading="lazy">` : ""}<span>${idLabel(i.id)} · severity ${esc(i.severity)}/5</span></button>`).join("")}</div></div><div class="order-decision"><strong class="${approved ? "status-approved" : rejected ? "status-rejected" : "status-pending"}">${status}</strong><p>${approved ? "Approved in this workspace. No repair has been dispatched." : rejected ? "This work order will not proceed." : pending && MargReview.validReview(state.agent) ? "Inspect the evidence and record your decision." : "Complete a valid review before making a decision."}</p>${o.decision_note ? `<p class="decision-note">${esc(o.decision_note)}</p>` : ""}${o.decided_at ? `<p>${esc(new Date(o.decided_at).toLocaleString())}</p>` : ""}${pending ? `<div class="decision-actions"><button class="button primary" data-decision="approve" data-order="${esc(o.work_order_id)}" ${isReadonly() || state.busy || !MargReview.validReview(state.agent) ? "disabled" : ""}>Approve</button><button class="button quiet" data-decision="reject" data-order="${esc(o.work_order_id)}" ${isReadonly() || state.busy || !MargReview.validReview(state.agent) ? "disabled" : ""}>Reject</button></div>${isReadonly() ? "<p>Read-only workspace</p>" : ""}` : ""}</div></article>`;
    })
    .join("");
}
const toolTitles = {
  model: "Review reasoning",
  inspect_roi: "Inspect uncertain evidence",
  compare_frames: "Compare frame observations",
  draft_work_order: "Draft a work order",
  request_human_approval: "Request human approval",
  dismiss_instance: "Dismiss unconfirmed damage",
  request_resurvey: "Request a new survey",
  finalize: "Complete review",
};
function describeStep(entry) {
  const out = entry.output || {};
  if (entry.tool === "model") return out.text || entry.model_text || "";
  if (entry.tool === "inspect_roi")
    return `Defect ${idLabel(entry.input?.instance_id)} · ${out.confirmed ? "confirmed on closer inspection" : "not confirmed on closer inspection"}`;
  if (entry.tool === "compare_frames")
    return `Defect ${idLabel(entry.input?.instance_id)} · ${Array.isArray(out.classes_seen) ? out.classes_seen.join(", ") : "frame evidence compared"}`;
  if (entry.tool === "draft_work_order")
    return out.title || out.reason || entry.input?.summary || "";
  if (entry.tool === "request_human_approval")
    return "A human decision is required before this work order can proceed.";
  if (entry.tool === "dismiss_instance")
    return `Defect ${idLabel(entry.input?.instance_id)} · ${out.reason || entry.input?.reason || "insufficient evidence"}`;
  if (entry.tool === "request_resurvey")
    return out.reason || entry.input?.reason || "";
  return out.summary || entry.input?.summary || "";
}
function renderTrace() {
  $("traceProvenance").textContent = providerLabel();
  const a = state.agent;
  const checks = a?.audit;
  let auditText = "";
  if (checks) {
    const values = Array.isArray(checks)
      ? checks
      : Array.isArray(checks.rules)
        ? checks.rules
        : Array.isArray(checks.checks)
          ? checks.checks
          : [];
    auditText = values.length
      ? `${values.filter((c) => c.passed ?? c.ok).length} of ${values.length} policy checks passed.`
      : "";
  }
  $("agentSummary").hidden = !a;
  $("agentSummary").innerHTML = a
    ? `<strong>${esc(a.summary || "Review record")}</strong><p>${esc(a.status ? titleFor(a.status) : "Recorded review")} · ${number(a.tool_call_count || state.trace.filter((e) => e.tool !== "model").length)} tool calls${auditText ? " · " + esc(auditText) : ""}</p>`
    : "";
  $("traceList").innerHTML = state.trace.length
    ? state.trace
        .map(
          (e, index) =>
            `<li><span class="timeline-number">${index + 1}</span><div class="timeline-content"><strong>${esc(toolTitles[e.tool] || titleFor(e.tool))}</strong><span class="latency">${number(e.latency_ms, 1)} ms</span><p>${esc(describeStep(e))}</p>${e.output?.crop_path ? `<img class="timeline-crop" src="${media("agent_crops", e.output.crop_path)}" alt="Re-inspected road damage region" loading="lazy">` : ""}<details><summary>Inspection details</summary><pre>${esc(JSON.stringify({ input: e.input, output: e.output }, null, 2))}</pre></details></div></li>`,
        )
        .join("")
    : empty(
        "No activity recorded",
        "Run a review to see each evidence inspection and policy decision here.",
      );
  $("traceExport").disabled = !state.trace.length;
}
function exportJson(filename, data) {
  downloadFile(filename, JSON.stringify(data, null, 2), "application/json");
}
function downloadFile(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("Report download requested.");
}
function clearJob() {
  clearTimeout(state.jobTimer);
  state.jobTimer = null;
  state.job = null;
  state.busy = false;
  sessionStorage.removeItem("marg-active-job");
  applyCapabilities();
}
function trackJob(job) {
  clearTimeout(state.jobTimer);
  state.job = { ...job };
  state.busy = true;
  state.jobFailures = 0;
  sessionStorage.setItem("marg-active-job", JSON.stringify(state.job));
  showJob(job);
  applyCapabilities();
  pollJob();
}
function showJob(job) {
  const done = ["succeeded", "done", "failed"].includes(job.status);
  $("jobPanel").hidden = false;
  $("jobPanel").classList.toggle("finished", done);
  $("jobPanel").classList.toggle("failed", job.status === "failed");
  const kind = job.kind === "survey" ? "Survey" : "Review";
  $("jobTitle").textContent =
    job.status === "failed"
      ? `${kind} could not complete`
      : done
        ? `${kind} complete`
        : `${kind} ${job.status === "running" || job.status === "processing" ? "in progress" : "queued"}`;
  $("jobDetail").textContent =
    job.status === "failed"
      ? (typeof job.error === "string" ? job.error : job.error?.message) ||
        "Check the input and retry. Your saved evidence remains available."
      : done
        ? "The latest evidence and review record are available. Select the survey in the sidebar to inspect it."
        : job.kind === "survey"
          ? "Processing video and extracting road evidence. You can keep exploring."
          : "Inspecting evidence and checking review policy. Existing results stay available until completion.";
  $("jobDismiss").hidden = !done;
}
async function pollJob() {
  const tracked = state.job;
  if (!tracked) return;
  try {
    const job = await api(
      tracked.status_url || `/api/jobs/${encodeURIComponent(tracked.job_id)}`,
    );
    if (state.job !== tracked) return;
    state.jobFailures = 0;
    Object.assign(tracked, job);
    showJob(job);
    if (["succeeded", "done", "failed"].includes(job.status)) {
      const id = job.survey_id;
      clearJob();
      await loadSurveys(state.id || id);
      if (job.status !== "failed")
        toast(
          job.kind === "survey"
            ? "Survey is ready to inspect."
            : "Review completed.",
        );
      return;
    }
  } catch (error) {
    if (state.job !== tracked) return;
    state.jobFailures++;
    if (
      error.status === 404 ||
      error.status === 401 ||
      state.jobFailures >= 5
    ) {
      clearJob();
      showJob({
        kind: tracked.kind,
        status: "failed",
        error:
          error.status === 404
            ? "This job is no longer available. Refresh the survey to check its latest result."
            : `Could not retrieve job status. ${error.message}`,
      });
      return;
    }
  }
  state.jobTimer = setTimeout(pollJob, 1800);
}
async function runReview() {
  if (!state.result || state.busy) return;
  $("notice").hidden = true;
  state.busy = true;
  applyCapabilities();
  try {
    const job = await api(
      `/api/surveys/${encodeURIComponent(state.id)}/run_agent?llm=${encodeURIComponent($("reviewMode").value)}`,
      { method: "POST" },
    );
    trackJob({ ...job, kind: "agent" });
  } catch (error) {
    state.busy = false;
    applyCapabilities();
    if (error.status === 409 && error.detail?.job_id)
      trackJob({ ...error.detail, survey_id: state.id, kind: "agent" });
    else notify(`Review could not start. ${error.message}`, true);
  }
}
function openUpload() {
  if (!state.caps?.uploads?.enabled || isReadonly()) return;
  $("uploadError").hidden = true;
  showDialog("uploadDialog");
}
function formError(id, message) {
  $(id).textContent = message;
  $(id).hidden = false;
}
async function uploadSurvey(event) {
  event.preventDefault();
  const video = $("videoFile").files[0],
    gpx = $("gpxFile").files[0];
  const caps = state.caps.uploads;
  if (!video) return;
  const extension = video.name.slice(video.name.lastIndexOf(".")).toLowerCase();
  if (!caps.accepted_video_extensions.includes(extension))
    return formError("uploadError", "Choose an MP4, MOV, AVI, MKV or WebM video.");
  if (gpx && !gpx.name.toLowerCase().endsWith(".gpx"))
    return formError("uploadError", "Choose a timestamped .gpx track.");
  if (video.size > caps.max_video_bytes)
    return formError(
      "uploadError",
      `Video exceeds the ${number(caps.max_video_bytes / 1048576)} MB limit.`,
    );
  if (gpx && gpx.size > caps.max_gpx_bytes)
    return formError(
      "uploadError",
      `GPS track exceeds the ${number(caps.max_gpx_bytes / 1048576)} MB limit.`,
    );
  $("submitUpload").disabled = true;
  $("submitUpload").textContent = "Uploading…";
  $("uploadError").hidden = true;
  try {
    const data = new FormData($("uploadForm"));
    if (!gpx) data.delete("gpx");
    const job = await api("/api/surveys/upload", {
      method: "POST",
      body: data,
    });
    $("uploadDialog").close();
    $("uploadForm").reset();
    $("videoLabel").textContent = "Choose a dashcam video";
    trackJob({ ...job, kind: "survey" });
    state.surveys = await api("/api/surveys");
    renderSurveyList();
    if (!state.result) await loadSurveys(job.survey_id);
    toast("Upload received. Processing your survey.");
  } catch (error) {
    formError("uploadError", error.message);
  } finally {
    $("submitUpload").disabled = false;
    $("submitUpload").innerHTML = `Process survey${icon("arrow")}`;
  }
}
function openDecision(orderId, decision) {
  const order = state.agent?.work_orders.find(
    (o) => o.work_order_id === orderId,
  );
  if (!order || isReadonly() || state.busy || !MargReview.validReview(state.agent)) return;
  state.decision = { orderId, decision, surveyId: state.id };
  $("decisionTitle").textContent =
    decision === "approve" ? "Approve work order" : "Reject work order";
  $("decisionSummary").textContent = order.title || order.summary;
  $("decisionNote").value = "";
  $("decisionError").hidden = true;
  $("submitDecision").textContent =
    decision === "approve" ? "Record approval" : "Record rejection";
  $("submitDecision").classList.toggle("danger", decision === "reject");
  showDialog("decisionDialog");
}
async function submitDecision(event) {
  event.preventDefault();
  const d = state.decision;
  if (!d) return;
  $("submitDecision").disabled = true;
  try {
    await api(
      `/api/surveys/${encodeURIComponent(d.surveyId)}/work_orders/${encodeURIComponent(d.orderId)}/decision`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          decision: d.decision,
          note: $("decisionNote").value.trim(),
        }),
      },
    );
    $("decisionDialog").close();
    if (state.id === d.surveyId) {
      state.agent = await api(
        `/api/surveys/${encodeURIComponent(d.surveyId)}/agent`,
      );
      updateMetrics();
      renderOrders();
      renderDefects();
      renderEvidence();
    }
    toast(
      d.decision === "approve"
        ? "Approval recorded. No repair has been dispatched."
        : "Rejection recorded.",
    );
  } catch (error) {
    formError("decisionError", error.message);
  } finally {
    $("submitDecision").disabled = false;
  }
}
function expandImage(src, title, caption) {
  $("fullImage").src = src;
  $("fullImage").hidden = false;
  $("fullImage").alt = title;
  $("imageDialogTitle").textContent = title;
  $("fullImageCaption").textContent = caption || "";
  showDialog("imageDialog");
}
async function boot() {
  try {
    state.caps = await api("/api/capabilities");
    const session = await api("/api/session");
    state.authenticated = session.authenticated;
    applyCapabilities();
    if (state.caps.authentication_required && !state.authenticated) {
      $("surveyList").innerHTML =
        '<p class="small muted">Unlock to view your surveys.</p>';
      notify("This workspace is protected. Unlock it to inspect your surveys.");
      showDialog("accessDialog");
      return;
    }
    await loadSurveys(new URLSearchParams(location.search).get("survey"));
    const saved = sessionStorage.getItem("marg-active-job");
    if (saved && !state.job) {
      try {
        const job = JSON.parse(saved);
        if (typeof job.job_id === "string") trackJob(job);
      } catch {
        sessionStorage.removeItem("marg-active-job");
      }
    }
  } catch (error) {
    notify(`Could not connect to your workspace. ${error.message}`, true);
    $("surveyList").innerHTML = empty(
      "Connection unavailable",
      "Use Refresh surveys to try again.",
    );
  }
}
document
  .querySelectorAll("[data-view]")
  .forEach((b) =>
    b.addEventListener("click", () => switchView(b.dataset.view)),
  );
document
  .querySelectorAll("[data-close]")
  .forEach((b) =>
    b.addEventListener("click", () => $(b.dataset.close).close()),
  );
$("surveyList").addEventListener("click", (event) => {
  const b = event.target.closest("[data-survey]");
  if (b) {
    selectSurvey(b.dataset.survey).catch((e) => notify(e.message, true));
    setSidebar(false);
  }
});
$("defectList").addEventListener("click", (event) => {
  const b = event.target.closest("[data-defect]");
  if (b) selectDefect(Number(b.dataset.defect));
});
$("frameStrip").addEventListener("click", (event) => {
  const b = event.target.closest("[data-frame]");
  if (b) showFrame(Number(b.dataset.frame));
});
$("orderList").addEventListener("click", (event) => {
  const b = event.target.closest("[data-decision]");
  if (b) openDecision(b.dataset.order, b.dataset.decision);
  const image = event.target.closest("[data-order-evidence]");
  if (image?.dataset.orderEvidence)
    expandImage(
      media("evidence", image.dataset.orderEvidence),
      image.dataset.evidenceTitle,
      "Work-order evidence · verify before recording a decision",
    );
});
$("showEvidence").addEventListener("click", () => {
  $("evidenceStage").hidden = false;
  $("mapStage").hidden = true;
  $("showEvidence").classList.add("active");
  $("showEvidence").setAttribute("aria-pressed", "true");
  $("showMap").classList.remove("active");
  $("showMap").setAttribute("aria-pressed", "false");
});
$("showMap").addEventListener("click", () => {
  $("evidenceStage").hidden = true;
  $("mapStage").hidden = false;
  $("showEvidence").classList.remove("active");
  $("showEvidence").setAttribute("aria-pressed", "false");
  $("showMap").classList.add("active");
  $("showMap").setAttribute("aria-pressed", "true");
  renderMap();
});
$("uncertainFilter").addEventListener("click", () => {
  state.uncertain = !state.uncertain;
  $("uncertainFilter").setAttribute("aria-pressed", String(state.uncertain));
  renderDefects();
});
$("nextStepAction").addEventListener("click", () => {
  const action = reviewState().next.action;
  if (action === "review") runReview();
  else {
    if (action === "orders") { $("orderFilter").value = "pending_approval"; renderOrders(); }
    switchView(action);
  }
});
$("outcomeFilter").addEventListener("change", renderDefects);
$("searchDefects").addEventListener("input", renderDefects);
$("sortDefects").addEventListener("change", renderDefects);
$("orderFilter").addEventListener("change", renderOrders);
$("newSurvey").addEventListener("click", openUpload);
$("emptyUpload").addEventListener("click", openUpload);
$("uploadForm").addEventListener("submit", uploadSurvey);
$("decisionForm").addEventListener("submit", submitDecision);
$("runReview").addEventListener("click", runReview);
$("refresh").addEventListener("click", async () => {
  try {
    if (!state.caps) await boot();
    else await loadSurveys();
    toast("Surveys refreshed.");
  } catch (e) {
    notify(e.message, true);
  }
});
$("helpButton").addEventListener("click", () => showDialog("helpDialog"));
const mobileQuery = matchMedia("(max-width:700px)");
function setSidebar(open) {
  const wasOpen = $("sidebar").classList.contains("open");
  $("sidebar").classList.toggle("open", open);
  $("menuButton").setAttribute("aria-expanded", String(open));
  $("sidebar").inert = mobileQuery.matches && !open;
  document.querySelector(".workspace").inert = mobileQuery.matches && open;
  $("navBackdrop").hidden = !mobileQuery.matches || !open;
  if (mobileQuery.matches && !open && wasOpen) $("menuButton").focus();
}
$("menuButton").addEventListener("click", () => {
  const open = !$("sidebar").classList.contains("open");
  setSidebar(open);
  if (open) $("sidebar").querySelector(".nav-item").focus();
});
$("navBackdrop").addEventListener("click", () => setSidebar(false));
mobileQuery.addEventListener("change", () => setSidebar(false));
setSidebar(false);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setSidebar(false);
  }
});
$("expandEvidence").addEventListener("click", () =>
  expandImage(
    $("evidenceImage").src,
    $("evidenceTitle").textContent,
    $("evidenceCaption").textContent,
  ),
);
$("evidenceImage").addEventListener("error", () => {
  $("evidenceImage").hidden = true;
  $("imageEmpty").hidden = false;
  $("imageEmpty").textContent =
    "This evidence image could not be loaded. Refresh the survey to try again.";
  $("expandEvidence").hidden = true;
});
$("videoFile").addEventListener("change", () => {
  $("videoLabel").textContent =
    $("videoFile").files[0]?.name || "Choose a dashcam video";
});
$("videoDrop").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("videoDrop").classList.add("dragging");
});
$("videoDrop").addEventListener("dragleave", () =>
  $("videoDrop").classList.remove("dragging"),
);
$("videoDrop").addEventListener("drop", (event) => {
  event.preventDefault();
  $("videoDrop").classList.remove("dragging");
  if (event.dataTransfer.files.length) {
    const transfer = new DataTransfer();
    transfer.items.add(event.dataTransfer.files[0]);
    $("videoFile").files = transfer.files;
    $("videoFile").dispatchEvent(new Event("change"));
  }
});
$("exportButton").addEventListener("click", () => {
  if (!state.result || state.busy) return;
  $("exportError").hidden = true;
  $("exportStatus").hidden = true;
  showDialog("exportDialog");
});
$("exportData").addEventListener("click", () => {
  if (!state.result) return;
  exportJson(`margai-${state.id}-report.json`, {
    exported_at: new Date().toISOString(),
    location_notice: state.result.geo_source === "synthetic"
      ? "Synthetic coordinates; do not use for dispatch." : "Field location verification required.",
    estimate_notice: "Area and severity are estimates; field verification required.",
    survey: state.result, review: state.agent,
  });
});
async function embeddedEvidence(url, remaining) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(url, { credentials: "same-origin", signal: controller.signal });
    const mime = response.headers.get("content-type")?.split(";")[0];
    if (!response.ok || !["image/jpeg", "image/png"].includes(mime)) return null;
    const limit = Math.min(2 * 1024 * 1024, remaining);
    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) { await reader.cancel(); return null; }
      chunks.push(value);
    }
    const blob = new Blob(chunks, { type: mime });
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
    return { data, size };
  } catch { return null; }
  finally { clearTimeout(timeout); }
}
$("exportReadable").addEventListener("click", async () => {
  if (!state.result || state.busy) return;
  const survey = structuredClone(state.result), review = structuredClone(state.agent);
  const surveyId = state.id, surveyName = $("surveyName").textContent;
  const outcomes = Object.fromEntries([...MargReview.summary(survey, review).outcomes].map(([id, outcome]) => [id, outcome.label]));
  $("exportReadable").disabled = true;
  $("exportData").disabled = true;
  $("exportError").hidden = true;
  $("exportStatus").hidden = false;
  $("exportStatus").textContent = "Preparing the report and embedding public evidence…";
  try {
    const images = {};
    let remaining = 8 * 1024 * 1024;
    for (const instance of survey.instances.slice(0, 12)) {
      if (!instance.evidence_path || remaining <= 0) continue;
      const image = await embeddedEvidence(media("evidence", instance.evidence_path, surveyId), remaining);
      if (image) { images[instance.id] = image.data; remaining -= image.size; }
    }
    const html = MargReport.buildReport({ survey, review, surveyId, surveyName,
      exportedAt: new Date().toISOString(), images, outcomes });
    downloadFile(`margai-${surveyId}-report.html`, html, "text/html");
    $("exportStatus").textContent = `Report download requested with ${Object.keys(images).length} embedded images. Open it in your browser to read or print to PDF. Missing images are labeled in the report.`;
  } catch (error) { formError("exportError", `Report could not be prepared. ${error.message}`); }
  finally { $("exportReadable").disabled = false; $("exportData").disabled = false; }
});
$("traceExport").addEventListener("click", () =>
  exportJson(`margai-${state.id}-activity.json`, {
    provider: state.agent?.provider || { kind: "unrecorded" },
    trace: state.trace,
  }),
);
$("jobDismiss").addEventListener("click", () => ($("jobPanel").hidden = true));
$("accessButton").addEventListener("click", async () => {
  if (state.authenticated) {
    try {
      await api("/api/session", { method: "DELETE" });
      clearJob();
      state.authenticated = false;
      state.result = null;
      state.surveys = [];
      state.id = null;
      $("surveyWorkspace").hidden = true;
      $("emptyWorkspace").hidden = true;
      renderSurveyList();
      applyCapabilities();
      notify("Workspace locked. Unlock to continue.");
    } catch (e) {
      notify(e.message, true);
    }
  } else showDialog("accessDialog");
});
$("accessForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("accessError").hidden = true;
  const token = $("accessToken").value;
  $("accessToken").value = "";
  try {
    await api("/api/session", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
    state.authenticated = true;
    $("accessDialog").close();
    $("notice").hidden = true;
    applyCapabilities();
    await loadSurveys();
  } catch (e) {
    formError("accessError", e.message);
  }
});
window.addEventListener("beforeunload", () => clearTimeout(state.jobTimer));
boot();
