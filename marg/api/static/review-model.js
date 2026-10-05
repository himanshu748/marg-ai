(function (root) {
  "use strict";
  const validReview = (review) =>
    review?.status === "completed" && review?.audit?.passed === true;

  function outcome(instance, review, survey) {
    if (!validReview(review)) return {
      code: "unreviewed", label: review ? "Fresh review required" : "Not reviewed",
      detail: "Run a successful policy review before recording a decision.",
      needsReview: instance.fused_conf < 0.55,
    };
    const dismissal = (review.dismissals || []).find((d) => d.instance_id === instance.id);
    if (dismissal) return {
      code: "dismissed", label: "Dismissed by review",
      detail: dismissal.reason || "Evidence did not confirm this detection. Kept for audit.",
      needsReview: false,
    };
    const segments = (survey.segments || []).filter((s) => s.instance_ids.includes(instance.id));
    const resurvey = (review.resurveys || []).find((r) => segments.some((s) => s.id === r.segment_id));
    if (resurvey) return {
      code: "resurvey", label: "Resurvey requested",
      detail: resurvey.reason || "Conflicting evidence needs a new field observation.", needsReview: true,
    };
    const order = (review.work_orders || []).find((o) => (o.instance_ids || []).includes(instance.id));
    if (order) return {
      code: "ordered",
      label: order.status === "approved" ? "Work order approved" : order.status === "rejected" ? "Work order rejected" : "Included in work order",
      detail: "Review the linked work order and its recorded decision. No repair is dispatched.", needsReview: false,
    };
    const confirmed = (review.trace || []).some((e) =>
      e.tool === "inspect_roi" && e.input?.instance_id === instance.id &&
      e.output?.confirmed === true && !e.output?.error);
    if (confirmed) return {
      code: "confirmed", label: "Confirmed on inspection",
      detail: "Re-inspection confirmed the detection; field verification is still required.", needsReview: false,
    };
    return instance.fused_conf < 0.55
      ? {code: "uncertain", label: "Evidence unresolved", detail: "Inspect review activity before deciding.", needsReview: true}
      : {code: "retained", label: "Retained evidence", detail: "No work order was required by the review policy.", needsReview: false};
  }

  function matchesOutcome(instance, review, survey, filter) {
    if (filter === "all") return true;
    if (filter === "ordered") return validReview(review) && (review.work_orders || []).some((o) => (o.instance_ids || []).includes(instance.id));
    return outcome(instance, review, survey).code === filter;
  }

  function summary(survey, review) {
    const outcomes = new Map((survey?.instances || []).map((i) => [i.id, outcome(i, review, survey)]));
    const valid = validReview(review);
    const orders = review?.work_orders || [];
    const pending = valid ? orders.filter((o) => o.status === "pending_approval").length : 0;
    const decided = valid ? orders.filter((o) => ["approved", "rejected"].includes(o.status)).length : 0;
    const dismissed = [...outcomes.values()].filter((o) => o.code === "dismissed").length;
    const unresolved = [...outcomes.values()].filter((o) => o.needsReview).length;
    const resurveys = valid ? (review.resurveys || []).length : 0;
    let next;
    if (!valid) next = {
      title: review ? "Run a fresh, checked review" : "Turn this evidence into a review",
      detail: review ? "This saved review is incomplete or has no passing policy audit. Its decisions are blocked." : "Inspect uncertainty, apply the repair policy, and prepare drafts for a human decision.",
      action: "review", label: review ? "Run fresh review" : "Run review",
    };
    else if (pending) next = {
      title: `${pending} work order${pending === 1 ? " needs" : "s need"} your decision`,
      detail: "Review the supporting evidence, then approve or reject each draft. Approval records a decision only.",
      action: "orders", label: "Review work orders",
    };
    else if (resurveys) next = {
      title: "New field evidence is needed", detail: `${resurveys} resurvey request${resurveys === 1 ? " is" : "s are"} recorded. Check the reason and plan another observation.`,
      action: "activity", label: "View resurvey reasons",
    };
    else if (unresolved) next = {
      title: "Inspect the unresolved evidence", detail: "Some low-confidence evidence has no recorded resolution. Check the activity before using this survey.",
      action: "activity", label: "View review activity",
    };
    else next = {
      title: "Review complete", detail: "Policy checks passed. Inspect the activity or export a report with the evidence and recorded decisions.",
      action: "activity", label: "View review activity",
    };
    return {valid, outcomes, pending, decided, dismissed, unresolved, resurveys, next};
  }
  const api = {validReview, outcome, matchesOutcome, summary};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MargReview = api;
})(typeof window === "undefined" ? {} : window);
