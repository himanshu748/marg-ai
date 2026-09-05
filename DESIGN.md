# MargAI — Agentic Road-Damage Survey (design)

Target: OpenCV AI Competition 2026 (AWS). Overall prize + Agentic Vision Award + Best Use of COOL Award.

## One-liner
Dashcam video + GPS in → OpenCV 5 pipeline turns it into de-duplicated, geo-located, severity-scored
road-damage instances → an agent inspects uncertain detections more closely (active perception),
requests re-surveys, drafts municipal work orders, and asks a human to approve before anything is filed.

## Repo layout
```
marg-ai/
  pyproject.toml            # python>=3.10; opencv-python==5.0.0.93, numpy, fastapi, uvicorn, boto3, pydantic, gpxpy, shapely(optional)
  marg/
    vision/
      ingest.py             # VideoSource: decode w/ cv.VideoCapture, resize, timestamp each frame
      quality.py            # blur (Laplacian variance), exposure (histogram), skip bad frames
      keyframes.py          # ORB/AKAZE features + homography inliers -> drop near-duplicate frames
      detector.py           # cv.dnn ONNX YOLO (RDD2022 classes D00,D10,D20,D40); letterbox, NMS (cv.dnn.NMSBoxes)
      classical.py          # classical pothole/crack cue: dark-blob + texture (Laplacian/LBP) + morphology -> support score fused w/ DNN conf
      tracker.py            # cross-frame instance association: IoU + Farneback flow propagation -> unique damage instances
      severity.py           # inverse-perspective mapping (homography from assumed road plane / lane calibration) -> area m^2; class+area -> severity 1..5
      geo.py                # GPX/SRT parse, time-align to frames, haversine, cluster instances (radius 25m) -> road segments
      pipeline.py           # run(video, gpx) -> SurveyResult (instances, segments, frames, metrics)
    agent/
      tools.py              # tool implementations (pure python, each returns JSON):
                            #   inspect_roi(frame_id, bbox, upscale)   -> re-run detector on 2x/3x crop (active perception)
                            #   compare_frames(instance_id)            -> confidence across frames of same instance
                            #   request_resurvey(segment_id, reason)
                            #   draft_work_order(segment_id, priority, summary)
                            #   request_human_approval(work_order_id)
                            #   finalize()
      loop.py               # perception -> decision -> action loop over Bedrock Converse API (tool use). Policy constraints:
                            #   * conf < 0.55 => MUST call inspect_roi before deciding
                            #   * class disagreement across frames => request_resurvey
                            #   * severity >= 4 => draft_work_order(priority=high) then request_human_approval
                            #   * max 25 tool calls / survey; every step appended to trace
      trace.py              # structured trace (step, tool, input, output, latency) -> JSONL + DynamoDB
      llm.py                # Bedrock client wrapper; offline `MockLLM` that follows the same policy deterministically (for tests/CI)
    store/
      base.py               # Store interface: surveys, instances, work_orders, traces
      local.py              # sqlite/json local store
      dynamo.py             # DynamoDB store
      blobs.py              # local fs / S3 for videos, keyframes, crops
    api/
      app.py                # FastAPI: POST /surveys (upload video+gpx), GET /surveys/{id}, GET /surveys/{id}/trace,
                            #          GET /work-orders, POST /work-orders/{id}/approve|reject, GET /health, GET /metrics
      worker.py             # SQS consumer (AWS) / in-process background task (local): runs pipeline + agent
      static/               # index.html: Leaflet map, instance list w/ crops, agent trace timeline, approve/reject buttons
  eval/
    eval_detector.py        # mAP@0.5 per class on RDD2022 India val split (COCO-style), latency
    eval_dedupe.py          # tracker: unique-instance precision/recall vs hand-labelled clip
    eval_agent.py           # scripted scenarios: does the agent inspect low-conf, escalate severe, ask approval? task success %
    bench_cool.py           # same pipeline: OpenCV pip (x86) vs COOL (Graviton) -> fps, latency, $/hour
  infra/
    Dockerfile              # multi-arch (amd64 + arm64); arm64 target uses COOL
    cdk/ or terraform/      # S3, SQS, DynamoDB, ECS Fargate (Graviton), ALB, IAM, CloudWatch, Bedrock access
  docs/
    architecture.md, report.md, video-script.md
  tests/
```

## Judging alignment
- Technical execution 30%: multi-stage OpenCV 5 use (dnn, features2d, calib3d homography, video/optflow, imgproc), tracker, IPM sizing.
- Innovation 20%: agentic active perception — vision confidence drives zoom re-inspection & resurvey; fusion of classical cue + DNN.
- Impact 20%: India road damage; RDD2022 India data; municipal work orders.
- UX 10%: map dashboard, approvals, trace timeline.
- Docs 10%: report, diagram, pinned deps, reproducible eval.
- Cloud 10%: Graviton Fargate, S3/SQS/DynamoDB, CloudWatch, IAM least-priv, responsible-use section.
