const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const { buildReport } = require("../marg/api/static/report.js");

const png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a8iAAAAAASUVORK5CYII=";
const defect = (id = 0) => ({ id, class_name: "pothole", fused_conf: 0, severity: 0, area_m2: 0, lat: 0, lon: 0 });
// Inspect attribute names after removing safely quoted values. Attack text in
// an escaped alt value is inert and must not be confused with an attribute.
const attributeNames = (html) => (html.match(/<[^>]*>/g) || [])
  .map((tag) => tag.replace(/"[^"]*"|'[^']*'/g, '""')).join("\n");
const snapshot = () => ({
  surveyName: "Route 7", surveyId: "survey-7", exportedAt: "2026-10-05T09:00:00Z",
  survey: { geo_source: "synthetic", instances: [defect()], keyframes: 0, processed_frames: 0 },
  review: { status: "completed", provider: { kind: "mock" }, audit: { passed: true, rules: [{ name: "Evidence checked", passed: true, offending_ids: [] }] }, run_id: "run-7", completed_at: "2026-10-05T08:00:00Z", work_orders: [] },
});

test("exports a browser global and a complete self-contained print document", () => {
  const context = vm.createContext({});
  vm.runInContext(fs.readFileSync(require.resolve("../marg/api/static/report.js"), "utf8"), context);
  assert.equal(typeof context.MargReport.buildReport, "function");
  const html = buildReport(snapshot());
  assert.match(html, /^<!doctype html>/);
  assert.match(html, /<html lang="en">/);
  assert.match(html, /@media print/);
  assert.match(html, /@page/);
  assert.doesNotMatch(html, /<script\b|<link\b|<iframe\b|\son\w+=|https?:\/\//i);
  assert.match(html, /Content-Security-Policy/);
});

test("escapes titles, classes, IDs, outcomes, notes, audit fields and provenance", () => {
  const attack = '<script>alert("x")</script><img src=x onerror=alert(1)> & \'injected\'';
  const input = snapshot();
  input.surveyName = attack;
  input.surveyId = attack;
  input.exportedAt = attack;
  input.survey.geo_source = attack;
  input.survey.video = attack;
  input.survey.instances = [{ ...defect(attack), class_name: attack }];
  input.outcomes = { [attack]: attack };
  input.review = { status: attack, summary: attack, run_id: attack, completed_at: attack,
    provider: { kind: attack, model_id: attack, region: attack },
    audit: { passed: false, rules: [{ name: attack, passed: false, offending_ids: [attack] }] },
    work_orders: [{ title: attack, reason: attack, work_order_id: attack, priority: attack, segment_id: attack, instance_ids: [attack], status: attack, decision_note: attack, decided_at: attack }],
  };
  const html = buildReport(input);
  assert.ok(!html.includes(attack));
  assert.match(html, /&lt;script&gt;alert\(&quot;x&quot;\)&lt;\/script&gt;/);
  assert.match(html, /&amp; &#39;injected&#39;/);
  assert.doesNotMatch(html, /<script\b|<img\b|<[^>]*\sonerror=/i);
  assert.match(html, /Fresh successful review needed/);
  input.images = { [attack]: png };
  const withImage = buildReport(input);
  assert.match(withImage, /alt="Evidence for defect &lt;script&gt;/);
  assert.doesNotMatch(attributeNames(withImage), /\son\w+=/i);
});

test("embeds only canonical PNG or JPEG base64, rejecting unsafe image sources", () => {
  for (const image of ["https://example.com/evidence.png", "javascript:alert(1)", "data:image/svg+xml;base64,PHN2Zz4=", "data:image/png;base64,", "data:image/png;base64,YQ", "data:image/png;base64,YR==", "data:image/png;base64,YQ==\n", 'data:image/png;base64,YQ==" onerror="alert(1)', "data:image/png;name=x;base64,YQ=="]) {
    const html = buildReport({ ...snapshot(), images: { 0: image } });
    assert.doesNotMatch(html, /<img\b/);
    assert.match(html, /Evidence image unavailable in this export/);
  }
  for (const image of [png, "data:image/jpeg;base64,/9j/2Q=="]) {
    const html = buildReport({ ...snapshot(), images: { 0: image } });
    assert.ok(html.includes(`src="${image}"`));
    assert.match(html, /1 of 1 defect images embedded/);
  }
});

test("requires explicit completed status and boolean passed audit for readiness", () => {
  for (const review of [null, {}, { status: "completed" }, { audit: { passed: true } }, { status: "completed", audit: { passed: "true" } }, { status: "failed", audit: { passed: true } }, { status: "completed", audit: { passed: false } }]) {
    const html = buildReport({ ...snapshot(), review });
    assert.match(html, /<strong>Fresh successful review needed<\/strong>/);
    assert.doesNotMatch(html, /<strong>Review completed · audit passed<\/strong>/);
  }
  assert.match(buildReport(snapshot()), /<strong>Review completed · audit passed<\/strong>/);
});

test("records caveats and missing provenance without invented values", () => {
  const html = buildReport(snapshot());
  assert.match(html, /Synthetic GPS coordinates are demonstration locations/);
  assert.match(html, /Do not use them for dispatch or navigation/);
  assert.match(html, /Area and severity are estimates/);
  assert.match(html, /No repair has been dispatched/);
  assert.match(html, /Deterministic mock review · no live model inference/);
  assert.match(html, /run-7/);
  assert.match(html, /2026-10-05 08:00:00 UTC/);
  const missing = buildReport({ survey: { instances: [defect()] } });
  assert.match(missing, /Provider provenance is not recorded/);
  assert.match(missing, /Location provenance is not recorded/);
  assert.match(missing, /No detailed audit checks recorded/);
  assert.match(missing, /Not recorded/);
});

test("retains zero values and handles an empty survey", () => {
  const html = buildReport(snapshot());
  assert.match(html, /<td class="mono">0<\/td><td>pothole<\/td><td>0%<\/td><td>0 \/ 5<\/td><td>0 m²<\/td>/);
  assert.match(html, /Synthetic GPS: 0, 0/);
  assert.match(html, /Keyframes \/ processed frames<\/dt><dd>0 \/ 0/);
  const empty = buildReport();
  assert.match(empty, /No defects were recorded/);
  assert.match(empty, /No work orders or decisions recorded/);
  assert.doesNotMatch(empty, /undefined|NaN/);
  assert.doesNotThrow(() => buildReport(null));
});

test("preserves every defect in the table and caps evidence cards at twelve", () => {
  const input = snapshot();
  input.survey.instances = Array.from({ length: 15 }, (_, index) => defect(index));
  input.outcomes = { 0: "Dismissed", 1: "Confirmed on inspection", 2: "Included in work order", 3: "Review required" };
  input.images = { 14: png };
  const html = buildReport(input);
  const register = html.match(/<table aria-label="Complete defect register">([\s\S]*?)<\/table>/)[1];
  assert.equal((register.match(/<tr>/g) || []).length, 16);
  assert.equal((html.match(/<article class="evidence-card">/g) || []).length, 12);
  assert.match(register, /<td class="mono">14<\/td>/);
  for (const outcome of Object.values(input.outcomes)) assert.ok(register.includes(outcome));
  assert.match(html, /Gallery limited to the first 12 defects/);
  assert.match(html, /0 of 15 defect images embedded/);
  assert.match(html, /15 evidence images unavailable/);
  assert.match(html, /Evidence image unavailable in this export/);
});

test("prints decision notes and dates, order counts and audit offenders", () => {
  const input = snapshot();
  input.review.work_orders = [
    { work_order_id: "wo-1", status: "approved", priority: "high", segment_id: 0, instance_ids: [0], title: "Repair pothole", reason: "Inspected", decision_note: "Checked with crew\nField visit required", decided_at: "2026-10-05T08:30:00Z" },
    { work_order_id: "wo-2", status: "rejected" },
    { work_order_id: "wo-3", status: "pending_approval" },
    { work_order_id: "wo-4", status: "draft" },
  ];
  input.review.audit.rules.push({ name: "Approval required", passed: false, offending_ids: ["wo-3"] });
  const html = buildReport(input);
  assert.match(html, /1 approved · 1 rejected · 1 awaiting approval · 1 with other/);
  assert.match(html, /Checked with crew\nField visit required/);
  assert.match(html, /2026-10-05 08:30:00 UTC/);
  assert.match(html, /<td>Approval required<\/td><td>Failed<\/td><td>wo-3<\/td>/);
});
