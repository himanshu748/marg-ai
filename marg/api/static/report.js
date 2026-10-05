(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.MargReport = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const list = (value) => Array.isArray(value) ? value : [];
  const record = (value) => value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const present = (value, fallback = "Not recorded") => value === undefined || value === null || value === "" ? fallback : value;
  const number = (value, digits = 0) => typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US", { maximumFractionDigits: digits }) : "Not recorded";
  const date = (value) => {
    if (value === undefined || value === null || value === "") return "Not recorded";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toISOString().replace("T", " ").replace(".000Z", " UTC");
  };
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);

  // Exact MIME allowlist, canonical base64 padding and pad bits; no whitespace,
  // URL, SVG, extra media parameters, or attribute delimiters can be accepted.
  function safeImage(value) {
    if (typeof value !== "string") return null;
    const match = /^data:image\/(?:jpeg|png);base64,((?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?)$/.exec(value);
    if (!match || !match[1]) return null;
    const encoded = match[1];
    const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (encoded.endsWith("==") && alphabet.indexOf(encoded.at(-3)) % 16 !== 0) return null;
    if (!encoded.endsWith("==") && encoded.endsWith("=") && alphabet.indexOf(encoded.at(-2)) % 4 !== 0) return null;
    return value;
  }

  function buildReport(input = {}) {
    input = record(input);
    const survey = record(input.survey);
    const review = record(input.review);
    const provider = record(review.provider);
    const audit = record(review.audit);
    const instances = list(survey.instances).map(record);
    const orders = list(review.work_orders).map(record);
    const checks = list(audit.rules ?? audit.checks).map(record);
    const images = record(input.images);
    const outcomes = record(input.outcomes);
    const ready = review.status === "completed" && audit.passed === true;
    const title = present(input.surveyName, "Road survey");
    const surveyId = present(input.surveyId ?? survey.survey_id);
    const imageFor = (instance) => own(images, String(instance.id)) ? safeImage(images[String(instance.id)]) : null;
    const outcomeFor = (instance) => own(outcomes, String(instance.id)) ? present(outcomes[String(instance.id)], "Not reviewed") : "Not reviewed";
    const imageCount = instances.slice(0, 12).filter((instance) => imageFor(instance)).length;
    const approved = orders.filter((order) => order.status === "approved").length;
    const rejected = orders.filter((order) => order.status === "rejected").length;
    const pending = orders.filter((order) => order.status === "pending_approval").length;
    const auditLabel = audit.passed === true ? "Passed" : audit.passed === false ? "Failed" : "Not recorded";
    const locationNotice = survey.geo_source === "synthetic"
      ? "Synthetic GPS coordinates are demonstration locations. Do not use them for dispatch or navigation."
      : survey.geo_source ? "Field location verification is required before planning repair or dispatch."
        : "Location provenance is not recorded. Field location verification is required before planning repair or dispatch.";
    const providerNotice = provider.kind === "mock"
      ? "Deterministic mock review · no live model inference."
      : provider.kind ? `Recorded provider: ${provider.kind}. Live inference: ${provider.live_inference === true ? "recorded as enabled" : provider.live_inference === false ? "recorded as disabled" : "not recorded"}.`
        : "Provider provenance is not recorded. Live model inference cannot be verified.";

    const rows = instances.map((instance) => `<tr><td class="mono">${escape(present(instance.id))}</td><td>${escape(present(instance.class_name))}</td><td>${typeof instance.fused_conf === "number" && Number.isFinite(instance.fused_conf) ? escape(number(instance.fused_conf * 100, 1)) + "%" : "Not recorded"}</td><td>${escape(number(instance.severity))}${typeof instance.severity === "number" && Number.isFinite(instance.severity) ? " / 5" : ""}</td><td>${escape(number(instance.area_m2, 2))}${typeof instance.area_m2 === "number" && Number.isFinite(instance.area_m2) ? " m²" : ""}</td><td>${escape(outcomeFor(instance))}</td></tr>`).join("");
    const cards = instances.slice(0, 12).map((instance) => {
      const image = imageFor(instance);
      return `<article class="evidence-card">${image ? `<img src="${escape(image)}" alt="Evidence for defect ${escape(present(instance.id))}">` : '<div class="missing-image">Evidence image unavailable in this export</div>'}<div class="evidence-body"><div class="eyebrow">Defect ${escape(present(instance.id))}</div><h3>${escape(present(instance.class_name, "Unclassified defect"))}</h3><p>${escape(outcomeFor(instance))}</p><p class="small">Estimated severity ${escape(number(instance.severity))} / 5 · estimated area ${escape(number(instance.area_m2, 2))} m²</p><p class="small">${survey.geo_source === "synthetic" ? "Synthetic GPS" : "Recorded coordinates"}: ${escape(number(instance.lat, 5))}, ${escape(number(instance.lon, 5))}</p></div></article>`;
    }).join("");
    const decisions = orders.map((order) => `<article class="decision"><div class="decision-top"><h3>${escape(present(order.title ?? order.summary, "Road repair work order"))}</h3><span class="badge">${escape(present(order.status, "Status not recorded"))}</span></div><p>${escape(present(order.reason ?? order.summary, "Reason not recorded"))}</p><dl class="details"><div><dt>Work order ID</dt><dd class="mono">${escape(present(order.work_order_id))}</dd></div><div><dt>Priority / segment</dt><dd>${escape(present(order.priority))} / ${escape(present(order.segment_id))}</dd></div><div><dt>Defect IDs</dt><dd>${escape(list(order.instance_ids).length ? order.instance_ids.join(", ") : list(order.instances).map((instance) => present(record(instance).id)).join(", ") || "Not recorded")}</dd></div><div><dt>Decision date</dt><dd>${escape(date(order.decided_at))}</dd></div></dl><div class="note"><strong>Decision note</strong><p>${escape(present(order.decision_note, "No decision note recorded."))}</p></div></article>`).join("");
    const auditRows = checks.map((check) => `<tr><td>${escape(present(check.name ?? check.rule, "Unnamed policy check"))}</td><td>${(check.passed ?? check.ok) === true ? "Passed" : (check.passed ?? check.ok) === false ? "Failed" : "Not recorded"}</td><td>${escape(list(check.offending_ids).length ? check.offending_ids.join(", ") : "None recorded")}</td></tr>`).join("");

    return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>${escape(title)} · MargAI survey report</title><style>
:root{color-scheme:light;--ink:#153c38;--muted:#586b68;--line:#d7e3df;--teal:#087e70;--paper:#fafbf8}*{box-sizing:border-box}body{margin:0;background:#edf3f0;color:var(--ink);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1100px;margin:32px auto;padding:48px;background:var(--paper);box-shadow:0 12px 60px #153c3810}header{padding-bottom:28px;border-bottom:2px solid var(--ink)}.masthead{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:30px}.brand{font-size:23px;font-weight:800;letter-spacing:-1px}.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:10px;font-weight:700;color:var(--teal)}h1{font-size:clamp(28px,5vw,48px);line-height:1.1;letter-spacing:-1.8px;margin:8px 0 18px;overflow-wrap:anywhere}h2{font-size:24px;letter-spacing:-.6px;margin:0}h3{font-size:16px;line-height:1.35;margin:5px 0 10px}p{margin:8px 0;overflow-wrap:anywhere}.small,dt{font-size:12px;color:var(--muted)}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.badge{display:inline-block;background:#e0efe9;color:var(--ink);padding:5px 10px;border-radius:4px;font-size:11px;font-weight:700;overflow-wrap:anywhere}.readiness{padding:16px 20px;background:#e5f2eb;border-left:4px solid var(--teal);margin:24px 0}.readiness.needs-review{background:#fbf0dc;border-color:#a77928}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric{border:1px solid var(--line);padding:18px}.metric strong{display:block;font-size:29px;line-height:1.2}.metric span{font-size:12px;color:var(--muted)}.notice{background:#fff;border:1px solid var(--line);padding:16px 20px}.notice p{font-size:12px}.section{margin-top:38px}.section-heading{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:14px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed}th{text-align:left;background:#e9f0eb;font-size:10px;letter-spacing:.04em;text-transform:uppercase;padding:11px 9px}td{padding:12px 9px;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}thead{display:table-header-group}.evidence-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.evidence-card{border:1px solid var(--line);background:#fff;break-inside:avoid}.evidence-card img{display:block;width:100%;height:165px;object-fit:contain;background:#f0f3ef}.missing-image{height:165px;display:flex;align-items:center;justify-content:center;text-align:center;padding:24px;background:#edf1ed;color:var(--muted);font-size:12px}.evidence-body{padding:16px}.evidence-body p{font-size:12px}.decision{padding:22px;border:1px solid var(--line);background:#fff;margin-bottom:14px;break-inside:avoid}.decision-top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}.details{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:18px 0}.details dt{margin-bottom:3px}.details dd{margin:0;font-size:12px;overflow-wrap:anywhere}.note{border-top:1px solid var(--line);padding-top:12px;font-size:12px;white-space:pre-wrap}.empty{padding:20px;background:#fff;border:1px solid var(--line);color:var(--muted);font-size:13px}footer{border-top:2px solid var(--ink);margin-top:38px;padding-top:18px;font-size:11px;color:var(--muted)}
@media(max-width:750px){main{margin:0;padding:24px}.metrics{grid-template-columns:repeat(2,1fr)}.evidence-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.masthead,.section-heading{align-items:flex-start}table{min-width:640px}.details{grid-template-columns:1fr}}@media(max-width:450px){.evidence-grid{grid-template-columns:1fr}.masthead{flex-direction:column}}
@page{size:A4;margin:14mm}@media print{body{background:#fff;font-size:11px;-webkit-print-color-adjust:exact;print-color-adjust:exact}main{max-width:none;margin:0;padding:0;box-shadow:none;background:#fff}h1{font-size:34px}h2{font-size:20px}.masthead{margin-bottom:18px}.section{margin-top:24px}.metrics{grid-template-columns:repeat(4,1fr);gap:8px}.metric{padding:12px}.metric strong{font-size:24px}.evidence-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.evidence-card img,.missing-image{height:150px}.table-wrap{overflow:visible}table{min-width:0;font-size:10px}th,td{padding:8px 6px}tr,.readiness,.notice,.metric{break-inside:avoid}h2,h3,.section-heading{break-after:avoid}.details{grid-template-columns:repeat(2,minmax(0,1fr))}footer{margin-top:24px}}
</style></head><body><main>
<header><div class="masthead"><div class="brand">MargAI<span class="eyebrow"> / Road intelligence</span></div><div class="small">Survey review report<br>Exported ${escape(date(input.exportedAt))}</div></div><div class="eyebrow">Evidence &amp; review record</div><h1>${escape(title)}</h1><p class="small">Survey ID <span class="mono">${escape(surveyId)}</span> · Location source: ${escape(present(survey.geo_source))}</p></header>
<div class="readiness${ready ? "" : " needs-review"}"><strong>${ready ? "Review completed · audit passed" : "Fresh successful review needed"}</strong><p class="small">${ready ? "The recorded review completed and its policy audit explicitly passed. Human decisions and field verification remain required." : "A completed review with an explicitly passed policy audit is required before this report can support work-order decisions."}</p><p class="small">Current review status: ${escape(present(review.status))} · Policy audit: ${auditLabel}</p></div>
<div class="metrics"><div class="metric"><strong>${instances.length}</strong><span>Recorded defects</span></div><div class="metric"><strong>${orders.length}</strong><span>Work orders</span></div><div class="metric"><strong>${approved}</strong><span>Approved in workspace</span></div><div class="metric"><strong>${pending}</strong><span>Awaiting approval</span></div></div>
<div class="notice"><strong>Read before acting</strong><p>${escape(locationNotice)}</p><p>Area and severity are estimates based on an assumed road plane; field verification is required.</p><p>No repair has been dispatched. Approval in this workspace records a decision only.</p><p>${escape(providerNotice)}</p></div>
<section class="section"><div class="section-heading"><h2>Review summary</h2><span class="eyebrow">01 / Assessment</span></div><p>${escape(present(review.summary, "No review summary recorded."))}</p><p class="small">Work orders: ${approved} approved · ${rejected} rejected · ${pending} awaiting approval · ${orders.length - approved - rejected - pending} with other or unrecorded status.</p></section>
<section class="section"><div class="section-heading"><h2>Complete defect register</h2><span class="eyebrow">02 / ${instances.length} defects</span></div><p class="small">Confidence reflects fused observations. Severity and area are estimated. Review outcomes are recorded labels.</p>${rows ? `<div class="table-wrap"><table aria-label="Complete defect register"><thead><tr><th>ID</th><th>Class</th><th>Confidence</th><th>Severity</th><th>Estimated area</th><th>Review outcome</th></tr></thead><tbody>${rows}</tbody></table></div>` : '<p class="empty">No defects were recorded in this survey.</p>'}</section>
<section class="section"><div class="section-heading"><h2>Evidence gallery</h2><span class="eyebrow">03 / Visual record</span></div><p class="small">${imageCount} of ${instances.length} defect images embedded. ${instances.length - imageCount} evidence images unavailable in this export. ${instances.length > 12 ? "Gallery limited to the first 12 defects; every defect appears in the complete register above." : "Each recorded defect is represented below."}</p>${cards ? `<div class="evidence-grid">${cards}</div>` : '<p class="empty">No defect evidence is available.</p>'}</section>
<section class="section"><div class="section-heading"><h2>Work orders &amp; decisions</h2><span class="eyebrow">04 / Human record</span></div>${decisions || '<p class="empty">No work orders or decisions recorded.</p>'}</section>
<section class="section"><div class="section-heading"><h2>Policy audit</h2><span class="badge">${auditLabel}</span></div>${auditRows ? `<div class="table-wrap"><table aria-label="Policy audit checks"><thead><tr><th>Policy check</th><th>Result</th><th>Offending IDs</th></tr></thead><tbody>${auditRows}</tbody></table></div>` : '<p class="empty">No detailed audit checks recorded. Fresh successful review needed if the completed review and passed audit are missing.</p>'}</section>
<section class="section"><div class="section-heading"><h2>Provenance</h2><span class="eyebrow">05 / Reproducibility</span></div><dl class="details"><div><dt>Review run ID</dt><dd class="mono">${escape(present(review.run_id))}</dd></div><div><dt>Review completion date</dt><dd>${escape(date(review.completed_at))}</dd></div><div><dt>Provider / model</dt><dd>${escape(present(provider.kind))} / ${escape(present(provider.model_id))}</dd></div><div><dt>Provider region</dt><dd>${escape(present(provider.region))}</dd></div><div><dt>Source video</dt><dd>${escape(present(survey.video))}</dd></div><div><dt>Keyframes / processed frames</dt><dd>${escape(number(survey.keyframes))} / ${escape(number(survey.processed_frames))}</dd></div></dl><p class="small">Missing provenance is shown as “Not recorded”. This export preserves the supplied review snapshot; confirm it is current before acting.</p></section>
<footer><strong>MargAI / Evidence-led road review</strong><p>Self-contained report. Use your browser’s Print command and choose Save as PDF. Embedded evidence is limited to validated PNG and JPEG data images.</p></footer>
</main></body></html>`;
  }

  return { buildReport };
});
