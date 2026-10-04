# MargAI — Devpost submission draft

## Project name

MargAI

## Tagline

Approval-gated road surveys from dashcam video

## Inspiration

Road damage is easy to see in aggregate and difficult to manage operationally.
An isolated photograph does not explain whether the same pothole was reported
five times, where it is on a route, or whether the evidence is strong enough
to justify a repair request. We wanted a field survey assistant that combines
computer vision with a transparent decision process instead of treating a
language-model response as an autonomous maintenance order.

## What it does

MargAI processes dashcam video and optional GPS to detect longitudinal cracks,
transverse cracks, alligator cracks, and potholes. It filters poor-quality
frames, selects keyframes, tracks detections into instances, estimates
severity and approximate area, and preserves the best full-resolution evidence.
An approval-gated agent can inspect uncertain regions, compare observations,
request a re-survey, dismiss weak instances, and draft municipal work orders.
The FastAPI/Leaflet dashboard shows the map, evidence, agent trace, and pending
approval actions.

## How we built it

The vision path uses OpenCV 5 Python APIs: `VideoCapture`, Laplacian quality
scoring, ORB and RANSAC homography keyframes, OpenCV DNN ONNX YOLOv8
inference, Farneback optical flow, perspective transforms for severity, and
classical LAB/texture cues. OpenCV Zoo YuNet and tiled LPD-YuNet provide
best-effort face and plate redaction at write time. Pydantic models define a durable survey contract.
The agent has explicit tools, a JSONL trace, a deterministic MockLLM for
reproducible tests, a Bedrock Converse wrapper, and a pure auditor. The local
dashboard serves JSON survey directories. The AWS target uses S3, SQS,
DynamoDB, ECR, ECS/Fargate Graviton, CloudWatch Logs, and an ALB, described by
CloudFormation and `infra/deploy.sh`.

## Challenges

The detector export initially had a row-width guard that skipped all RDD
outputs; the regression test now covers the four-box-plus-four-class layout.
Detections also occurred between keyframes, so evidence had to be saved at the
best processed observation rather than defaulting to the first keyframe.
Tight crops lost road context during agent inspection, which led to a
context-window rerun and IoU-aware confirmation rule. The official India
archive was inaccessible from the evaluation environment, so a derived
Hugging Face slice was used and its licensing caveat was recorded explicitly.
PR #5 recorded a September 7, 2026 AWS deployment, later removed for budget
control. The stack and `infra/deploy.sh` remain a future deployment path;
there is no current live endpoint claimed here.

## Accomplishments

The local RDD2022 India evaluation reached 0.7390 mAP@0.5, with 0.8376
precision at the production threshold. The two pothole demos measured
5.25:1 and 32.00:1 detection-to-instance compression. The agent scenarios
pass the auditor policy checks, including the 25-call budget case. The
dashboard exposes evidence, trace, work-order details, and approval decisions
instead of only returning a model score.

## What we learned

Temporal evidence is as important as per-frame confidence. A detector box can
be correct while its keyframe association is wrong, and a crop can be too
tight for a second detector pass. We also learned that agent reliability is
more measurable when policy rules are represented as tools and audit checks.
The evaluation failures make the next model improvements concrete: shadows,
haze, glare, vehicles, and small distant defects.

## What's next

Before operational use, MargAI needs stronger privacy recall on small plates
and non-frontal faces, calibrated physical severity, real GPS and map-matching
validation, official dataset-license clarification, broader hard-negative
evaluation, and a matched ARM64/x86 benchmark. Future deployment requires
separate approval, budget review, and fresh end-to-end verification.

## Built with

- OpenCV 5
- Python
- OpenCV DNN
- YOLOv8 ONNX
- NumPy
- FastAPI
- Leaflet
- Amazon S3
- Amazon SQS
- Amazon DynamoDB
- Amazon ECS / AWS Fargate Graviton
- Amazon Bedrock Converse API
- AWS CloudFormation

## Featured paths

- **Agentic Vision:** included. The agent inspects evidence, applies explicit
  policy rules, writes a trace, and gates work orders behind human approval.
- **Graviton:** the template targets an ARM64 Fargate task. `docs/BENCHMARK.md`
  preserves historical stock OpenCV 5 measurements from unmatched x86 and
  Graviton4 hosts. Those runs establish neither Fargate cost nor a COOL result.
  COOL was not run; current Marketplace access was not checked.

## Submission checklist

- [x] Technical report: `docs/TECHNICAL_REPORT.md`
- [x] Source repository and implementation
- [x] Architecture diagram: `docs/ARCHITECTURE.md`,
  `docs/architecture.svg`, and rendered `docs/architecture.png`
- [x] Local build, deploy, and test instructions in `README.md`
- [ ] Live endpoint: historical AWS stack removed for budget control
- [ ] Demonstration video, maximum five minutes: record using
  `docs/VIDEO_SCRIPT.md`
- [x] Evaluation evidence: detector, deduplication, agent tables, and eight
  annotated failure cases
- [x] License and attribution notes
