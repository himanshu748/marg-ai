const test = require('node:test');
const assert = require('node:assert/strict');
const {summary, validReview} = require('../marg/api/static/review-model.js');
const survey = {instances: [{id: 0, fused_conf: .4}, {id: 1, fused_conf: .45}, {id: 2, fused_conf: .8}], segments: [{id: 4, instance_ids:[0,1,2]}]};
const checked = {status:'completed', audit:{passed:true}, trace:[], dismissals:[], work_orders:[], resurveys:[]};
test('legacy or failed reviews cannot claim decision readiness', () => {
  for (const r of [null, {}, {status:'completed'}, {...checked, audit:{passed:'true'}}, {...checked,status:'failed'}]) {
    assert.equal(validReview(r), false);
    const s = summary(survey, {...r, work_orders:[{status:'pending_approval'}]});
    assert.equal(s.pending, 0);
    assert.equal(s.next.action, 'review');
  }
});
test('confirmed and dismissed low-confidence evidence is resolved without hiding it', () => {
  const r = {...checked, dismissals:[{instance_id:0,reason:'unconfirmed'}], trace:[{tool:'inspect_roi',input:{instance_id:1},output:{confirmed:true}}]};
  const s = summary(survey, r);
  assert.equal(s.outcomes.size, 3);
  assert.equal(s.dismissed, 1);
  assert.equal(s.unresolved, 0);
  assert.equal(s.outcomes.get(0).code, 'dismissed');
  assert.equal(s.outcomes.get(1).code, 'confirmed');
  assert.equal(s.outcomes.get(2).code, 'retained');
});
test('failed inspection does not count as confirmation and resurvey remains unresolved', () => {
  let s = summary(survey, {...checked, trace:[{tool:'inspect_roi',input:{instance_id:1},output:{confirmed:true,error:'failed'}}]});
  assert.equal(s.outcomes.get(1).needsReview, true);
  s = summary(survey, {...checked, resurveys:[{segment_id:4,reason:'classes disagree'}]});
  assert.equal(s.unresolved, 3);
  assert.equal(s.next.label, 'View resurvey reasons');
});
test('next action follows pending decisions, then completed decisions', () => {
  const r = {...checked,dismissals:[{instance_id:0},{instance_id:1}],work_orders:[{instance_ids:[2],status:'pending_approval'}]};
  assert.equal(summary(survey,r).next.action,'orders');
  r.work_orders[0].status='approved';
  const s = summary(survey,r);
  assert.equal(s.pending,0);
  assert.equal(s.decided,1);
  assert.equal(s.next.title,'Review complete');
});
test('empty survey has no invented work and still requires a review', () => {
  assert.equal(summary({instances:[],segments:[]},null).next.action,'review');
  assert.equal(summary({instances:[],segments:[]},checked).outcomes.size,0);
});


test("resurvey evidence remains discoverable in linked work orders", () => {
  const { matchesOutcome } = require("../marg/api/static/review-model.js");
  const instance = { id: 0, fused_conf: 0.8 };
  const survey = { instances: [instance], segments: [{ id: 4, instance_ids: [0] }] };
  const review = { status: "completed", audit: { passed: true },
    resurveys: [{ segment_id: 4 }], work_orders: [{ instance_ids: [0], status: "pending_approval" }] };
  assert.equal(matchesOutcome(instance, review, survey, "resurvey"), true);
  assert.equal(matchesOutcome(instance, review, survey, "ordered"), true);
  assert.equal(matchesOutcome(instance, { ...review, status: "failed" }, survey, "ordered"), false);
});
