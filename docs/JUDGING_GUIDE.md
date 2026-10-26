# Judge MargAI without live inference

**Award opt-in: Agentic Vision Award.** The Devpost Special Award
Consideration field was saved as Agentic Vision Award on 26 October 2026.
Devpost confirmed that the existing entry remains Submitted.

## Hosted recorded path

Open [the Kumasi sample survey](https://margai.13.204.212.172.sslip.io/?survey=pothole_kumasi).
The public workspace was checked on 26 October 2026 without login or model calls.

1. **Survey review:** inspect the recorded evidence, three tracked defects from
   96 detections, confidence, estimated area and synthetic-location labels.
2. **Review activity:** inspect the recorded deterministic review. The hosted
   Kumasi record shows eight tool calls and six of six policy checks passed.
   It inspects uncertain evidence, compares observations, dismisses an
   unconfirmed instance, drafts work orders, requests human approval and ends.
3. **Work orders:** inspect the linked evidence and approval state. No repair
   is filed or dispatched. The public recorded workspace is read-only.
4. **Export activity** and **Export report** retain the recorded trace and
   report for inspection. They do not request model inference.

The prominent **Open live review** / **Run live review** controls are optional.
Do not use them for the recorded-evidence walkthrough. They invoke a paid model
and may become unavailable when the allowance is exhausted. A recorded trace
is evidence of that earlier run, not a fresh live-model result.

## Repository fallback

These checked-in fixtures are earlier recorded runs. Their contents may differ
from the hosted records above; read the actual files rather than assuming the
hosted tool-call count or approval state applies to them.

| Evidence | Repository artifact |
| --- | --- |
| Kumasi survey and observations | [result.json](../demo/pothole_kumasi/result.json) |
| Kumasi recorded policy result | [agent_result.json](../demo/pothole_kumasi/agent/agent_result.json) |
| Kumasi recorded tool trace | [trace.jsonl](../demo/pothole_kumasi/agent/trace.jsonl) |
| Kumasi evidence images | [evidence/](../demo/pothole_kumasi/evidence/) |
| Cars sample and observations | [result.json](../demo/pothole_cars/result.json) |
| Cars recorded tool trace | [trace.jsonl](../demo/pothole_cars/agent/trace.jsonl) |
| New detector timing, 26 October | [fresh latency table](FRESH_LATENCY.md) |
| Historical detector accuracy | [600-image September evaluation](../eval/results/detector_india.md) |
| Agent workflow | [architecture](ARCHITECTURE.md) |
| Failure cases and limits | [failures/](failures/) |

## Local read-only fallback

After the README's dependency installation, run from the repository root:

```bash
env -u MARG_S3_BUCKET -u MARG_SQS_QUEUE_URL \
  MARG_READ_ONLY=1 MARG_AGENT_LLM=mock MARG_BEDROCK_ENABLED=0 \
  .venv/bin/python -m marg.api --data demo --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/?survey=pothole_kumasi`. This serves the existing
fixtures. No new survey processing or agent request is needed. The web assets,
including fonts and Leaflet, are vendored; optional map tiles require network.

## Judging-window operations

Point-in-time health and browser checks cannot guarantee uptime through
9 November. Before the cutoff, the host operator should confirm service restart
on reboot, sufficient disk space, backup of recorded surveys, certificate
renewal and retention through the end of judging. Preserve the recorded route
when live-review credits are depleted. The current public host is AWS Lightsail
in `ap-south-1`; the September Fargate stack is historical.

The reviewed Modal implementation and evidence workflow are included in this
checkout. On 26 October, host inspection confirmed restart-unless-stopped,
persistent data and TLS volumes, zero container restarts and 50.6 GB free disk.
Future availability still depends on retaining the existing host through judging.
See [video verification](VIDEO_VERIFICATION.md) for the actual saved video's
team credit and results timing; no replacement video has been claimed.
