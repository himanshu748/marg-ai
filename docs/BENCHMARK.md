# Graviton / OpenCV benchmark

Both runs use the same harness (`python -m eval.bench_cool run`: one warm-up, then two timed
end-to-end pipeline runs per demo video) and the same OpenCV 5.0.0 `opencv-python` wheel for
the respective architecture. Raw JSON: `eval/results/bench_x86_pip_opencv.json` and
`eval/results/bench_graviton_pip_opencv.json`.

| | x86 baseline | Graviton4 |
| --- | --- | --- |
| Host | Intel Xeon Platinum 8375C @ 2.90 GHz, 2 logical CPUs | AWS EC2 `c8g.large` (Neoverse-V2 / Graviton4 @ 2.8 GHz), 2 vCPUs |
| OpenCV | 5.0.0 `opencv-python` (x86_64 wheel), 2 threads, pthreads | 5.0.0 `opencv-python` (aarch64 wheel), 2 threads, pthreads |
| CPU features | baseline `SSE SSE2 SSE3` + dispatched `SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX` | baseline `NEON FP16`, dispatched `NEON_DOTPROD NEON_FP16 NEON_BF16`; custom HAL `carotene + KleidiCV 26.03` |
| COOL build marker | no | no |

## End-to-end pipeline

| Video | Arch | Source FPS | Processed FPS | Median wall (s) | Keyframes | Cost / 1,000 source frames |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cars_moving_into_pothole (621 frames) | x86_64 | 7.70 | 2.57 | 80.65 | 14 | $0.001781 |
| cars_moving_into_pothole (621 frames) | aarch64 | 4.53 | 1.51 | 137.23 | 14 | $0.002425 |
| pothole_kumasi (420 frames) | x86_64 | 7.32 | 2.44 | 57.42 | 15 | $0.001875 |
| pothole_kumasi (420 frames) | aarch64 | 4.33 | 1.44 | 96.93 | 14 | $0.002532 |

## Where the time goes (cars video, seconds per run)

| Stage | x86_64 | aarch64 | Ratio |
| --- | ---: | ---: | ---: |
| decode | 5.1 | 4.6 | 0.9x |
| quality gate | 0.5 | 0.5 | 1.0x |
| keyframes (ORB + matching) | 6.9 | 7.4 | 1.1x |
| **DNN detector** (YOLO ONNX, `cv2.dnn`) | **48.0** | **106.4** | **2.2x** |
| tracking | 0.0 | 0.0 | - |
| privacy redaction (YuNet + LPD) | 5.3 | 5.4 | 1.0x |

Take-aways:

- The classic OpenCV stages (decode, blur/Laplacian quality gate, ORB keyframe selection,
  YuNet-based redaction) run at parity on 2 Graviton4 vCPUs vs 2 Xeon vCPUs — the aarch64 wheel
  already ships the KleidiCV/carotene HAL for `imgproc`.
- The only stage that regresses is `cv2.dnn` inference of the ~230 ms YOLO detector: 2.2x
  slower on the stock aarch64 wheel. Because detection is ~60-75 % of wall time, the whole
  pipeline ends up 0.59x, and Fargate's 20 % ARM discount does not recover it
  ($0.0024 vs $0.0018 per 1,000 frames).
- This is exactly the gap the Cloud-Optimized OpenCV Library (COOL) targets on Graviton.
  We could not run COOL: it is only distributed as an AWS Marketplace AMI and subscribing was
  blocked on this account (payment method not yet verified). The harness already supports it —
  launch a COOL AMI, activate `/opt/cool/venvs/python_3.*`, and run
  `python -m eval.bench_cool run --label graviton_cool ...`, then `compare` against
  the JSON files above. Until then no COOL number is claimed anywhere in this repo.

## Detector micro-benchmark

x86_64, first **200** RDD2022-India images after one warm-up: median **229.55 ms**,
p95 **236.09 ms**, **4.34 images/s**. The Graviton run of this micro-benchmark is missing
(the image bundle shipped to the instance contained dangling symlinks, so 0 images loaded);
the per-stage `stage_detect_s` numbers above are the Graviton detector evidence.

## Cost model

Hourly us-east-1 Linux Fargate list rates for 1 vCPU + 2 GB: x86
`$0.04048 + 2 × $0.004445 = $0.04937/h`, ARM `$0.03238 + 2 × $0.00356 = $0.03950/h`
([AWS Fargate pricing](https://aws.amazon.com/fargate/pricing/)). Storage, data transfer and
public IPv4 are excluded. The Graviton benchmark itself ran on one self-terminating
`c8g.large` on-demand instance for 13 minutes (< $0.05).
