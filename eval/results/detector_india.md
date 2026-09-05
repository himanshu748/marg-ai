# RDD2022 India detector evaluation

- Images: **600**; ground-truth boxes: **1279**
- AP: **all-point interpolated AP over confidence-ranked detections**
- Production threshold: **0.25**; IoU threshold: **0.50**
- CPU: **INTEL(R) XEON(R) PLATINUM 8559C**

| Class | AP@0.5 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| D00 | 0.6819 | 156 | 44 | 91 |
| D10 | 0.7033 | 4 | 2 | 7 |
| D20 | 0.8593 | 307 | 37 | 61 |
| D40 | 0.7116 | 425 | 90 | 228 |
| **mAP** | **0.7390** | | | |

| Production metric | Value |
| --- | ---: |
| Precision @ 0.25 | 0.8376 |
| Recall @ 0.25 | 0.6974 |
| Median inference latency (ms) | 177.64 |
| P95 inference latency (ms) | 225.09 |
| Inference throughput (images/s) | 5.37 |

Failure images are in `docs/failures/`; predictions are red and ground truth is green.
