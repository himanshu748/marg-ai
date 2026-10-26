Existing recording checked on 26 October: see [actual video verification](VIDEO_VERIFICATION.md). The script below remains a proposed replacement, not a completed recording.

# Results-first MargAI demo script

This is a replacement recording plan. It does not verify the current saved
3:14.6 submission video. Before replacing that video, watch it anonymously from
start to finish and confirm every team member appears. Do not invent a team
member or present a prerecorded run as live inference.

## First minute

| Time | Picture | Spoken point |
| --- | --- | --- |
| 0:00-0:10 | Every team member on camera, with name and actual contribution | Each member introduces themselves. A solo entrant appears and introduces themselves. |
| 0:10-0:25 | Kumasi survey: 96 frame detections, three retained instances; select evidence | MargAI compresses repeated road-damage sightings into evidence a person can review. Location is synthetic and area is estimated. |
| 0:25-0:40 | Review activity: dismissal after inspection, work-order draft, approval request | Visual evidence changes what happens next. An unconfirmed detection is dismissed and actionable damage becomes an approval-gated draft. |
| 0:40-0:55 | Fresh latency table beside the historical accuracy table | State the 26 October median and p95 detector timings from `FRESH_LATENCY.md`. Clearly label September accuracy as historical. |
| 0:55-1:00 | Work orders, with approval gate and evidence | This is a recorded deterministic review. It needs no live inference credits and never dispatches repairs. |

## 1:00-1:40: show the actual workflow

Open another defect, compare keyframes, then show the recorded tool trace.
Point to the OpenCV output that caused an inspection, dismissal, resurvey or
work-order action. Use actual artifact contents; do not read invented counts.
Distinguish deterministic policy orchestration from any separately recorded
live model trace. No paid review is needed for this walkthrough.

## 1:40-2:15: show AWS delivery and responsible control

Show the HTTPS dashboard and health response, then the architecture. Identify
the current Lightsail deployment separately from the old Fargate template.
Show the human approval gate, read-only public workspace, synthetic GPS label,
privacy limitations and the fact that approval does not dispatch a repair.

## 2:15-2:50: show failure handling and reproducibility

Show a difficult detection, the recorded audit and repository fallback.
Explain that the new latency run measures existing demo keyframes, not fresh
held-out accuracy. Show the pinned dependencies and reproduction command.

## 2:50-3:15: close with the outcome and award path

Return to the evidence and repair draft. State: "We opt into the Agentic Vision
Award: perception changes subsequent inspections and actions, and a person
retains the final decision." Point to the public source and recorded-evidence
guide. Keep the full recording below the competition's five-minute limit.

## Release verification

- Every team member visibly appears and gives their real contribution.
- Results are visible before 1:00.
- The actual video link plays anonymously from start to finish.
- The video saved on the hackathon entry is the intended version.
- The optional award selection is saved in the submission form.

These are checks to perform, not claims that they have already passed.
