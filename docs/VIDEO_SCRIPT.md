# MargAI five-minute demo script

This script demonstrates the local, reproducible path. It does not imply that
the AWS stack is currently live.

## 0:00–0:25 — Problem and architecture

**On screen:** Open `docs/architecture.png`, then show the dashboard home page.

**Voice-over:** “MargAI turns dashcam footage into an auditable road-damage
survey. OpenCV 5 detects and tracks damage, an agent investigates uncertain
cases, and a human approves any municipal work order.”

```bash
cd /home/ubuntu/repos/marg-ai
```

## 0:25–1:05 — Run the vision pipeline

**On screen:** Terminal running the pipeline, then the output directory.

**Voice-over:** “The pipeline reads video timing and optional GPS, filters
poor-quality frames, selects stable keyframes, detects the four RDD2022
classes, tracks observations, estimates severity, and saves evidence. Its
OpenCV 5 YuNet privacy pass tiles the frame for small plates and pixelates
detected faces and plates without touching the damage bbox.”

```bash
/home/ubuntu/venv/bin/python -m marg.vision.pipeline \
  --video /home/ubuntu/assets/cars_moving_into_pothole_CC_BY_SA_4.0.webm \
  --out outputs/pothole_cars
```

## 1:05–1:35 — Start the agent

**On screen:** Agent trace JSONL and the generated agent result.

**Voice-over:** “The local MockLLM follows the same policy shape as the
Bedrock wrapper. It inspects low-confidence cases, compares inconsistent
frames, drafts work orders, requests approval for high and medium priority,
and finalizes within a 25-tool-call budget.”

```bash
/home/ubuntu/venv/bin/python -m marg.agent.loop \
  --result outputs/pothole_cars/result.json \
  --llm mock \
  --out outputs/pothole_cars/agent
```

## 1:35–2:25 — Explore the dashboard

**On screen:** Survey selector, map, instance table, evidence thumbnail, and
the Agent trace tab.

**Voice-over:** “Each tracked instance has a confidence, class, severity,
location, observation history, and a best-observation evidence frame. The
trace makes the agent's tool calls reviewable rather than hiding them behind
a single answer.”

```bash
/home/ubuntu/venv/bin/python -m marg.api \
  --data outputs \
  --port 8000
```

Open `http://localhost:8000`, select `pothole_cars`, and show the Instances
and Agent trace tabs.

## 2:25–3:05 — Review and approve a work order

**On screen:** Work orders tab, evidence thumbnail, reason, severity, and
OpenStreetMap link.

**Voice-over:** “The agent drafts a repair recommendation with affected
instances, evidence, area, severity, GPS, and a reason. It cannot approve the
work itself. The operator can inspect the evidence and approve or reject the
draft.”

Click **Approve** on a pending work order and show the persisted status.

## 3:05–3:45 — Show the evaluation evidence

**On screen:** Detector and deduplication Markdown reports.

**Voice-over:** “The detector evaluation uses 600 labelled India images.
All-point AP at IoU 0.5 is 0.7390 mAP. At the production threshold, precision
is 0.8376 and recall is 0.6974. The demos show why tracking matters: 96
detections become three tracked instances in the Kumasi clip.”

```bash
cat eval/results/detector_india.md
cat eval/results/dedupe.md
```

## 3:45–4:20 — Show failure cases and policy tests

**On screen:** `docs/failures/` thumbnails, then the agent evaluation table.

**Voice-over:** “The failure cases expose glare, haze, shadows, clutter, and
vehicle occlusion. The responsible response is not to hide these failures:
the dashboard keeps evidence visible and the approval gate remains in place.”

```bash
cat docs/failures/README.md
/home/ubuntu/venv/bin/python eval/eval_agent.py --llm mock
```

## 4:20–5:00 — Close with reproducibility and limitations

**On screen:** README local quickstart and CloudFormation file.

**Voice-over:** “The same data contract is prepared for S3, SQS, DynamoDB,
Fargate Graviton, and an ALB. AWS deployment is pending account activation,
so the local results are the verified evidence today. Future work includes
stronger privacy recall on small or distant plates, real GPS validation,
calibrated severity, and a live Graviton benchmark.”

```bash
/home/ubuntu/venv/bin/ruff check .
/home/ubuntu/venv/bin/python -m pytest -q
```
