# Fresh detector latency measurement, 26 October 2026

This is a CPU latency check on existing committed demonstration images, not a
new held-out detector-quality evaluation. September accuracy measurements remain
historical and are not superseded by this run.

| Measurement | Result |
| --- | ---: |
| Unique sample keyframes | 29 |
| Timed detector calls | 87 (3 passes) |
| Warm-up calls, excluded | 3 |
| Median detector latency | 368.97 ms |
| P95 detector latency | 400.00 ms |
| Sequential throughput | 2.67 images/s |

Environment: AMD EPYC 9V74 80-Core Processor, x86_64, Python
3.12.14, OpenCV 5.0.0
(`opencv-python-headless==5.0.0.93`), one OpenCV thread. Measured at
`2026-10-26T06:01:33.038381+00:00`. The timing includes letterboxing,
blob creation, DNN inference, thresholding and NMS. It excludes disk reads,
model loading, privacy redaction, agent calls and network latency. The run was
repeated after the test suite finished to avoid concurrent test load.

Reproduce from the repository root:

```bash
.venv/bin/python -m eval.bench_fixture_latency \
  --repeats 3 --threads 1 \
  --out /tmp/margai-fixture-latency.json
```

[Raw measurements](../eval/results/fixture_latency_2026-10-26.json) contain every
timing, input and model SHA-256, configuration, environment, detection counts
and the source commit. The script is [here](../eval/bench_fixture_latency.py).
The benchmark script was added after the recorded source commit; the detector,
model and input files are unchanged from that commit.

These are existing demo keyframes without ground-truth labels. This run does
not measure mAP, precision, recall, AWS performance, Modal latency, COOL,
or end-to-end survey throughput. Hardware and measurement conditions differ
from September, so these numbers do not establish a performance regression
or improvement. Shared-host timings vary.
