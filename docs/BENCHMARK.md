# Graviton / OpenCV benchmark

These are historical September 2026 measurements on different hosts, not a
matched Fargate comparison or evidence of COOL use. No benchmark was rerun for
the local readiness repair.

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

| Video | Arch | Source FPS | Processed FPS | Median wall (s) | Keyframes |
| --- | --- | ---: | ---: | ---: | ---: |
| cars_moving_into_pothole (621 frames) | x86_64 | 7.70 | 2.57 | 80.65 | 14 |
| cars_moving_into_pothole (621 frames) | aarch64 | 4.53 | 1.51 | 137.23 | 14 |
| pothole_kumasi (420 frames) | x86_64 | 7.32 | 2.44 | 57.42 | 15 |
| pothole_kumasi (420 frames) | aarch64 | 4.33 | 1.44 | 96.93 | 14 |

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

- In these runs, the listed classic OpenCV stages took similar time on the two
  hosts. The recorded aarch64 build includes the KleidiCV/carotene HAL for `imgproc`.
- The recorded DNN stage took about 2.2x as long on the Graviton host and the
  end-to-end source throughput was about 0.59x the x86 host. Hardware, allocation,
  and build differences prevent attributing these ratios solely to architecture.
- COOL was not run. The historical attempt reported a blocked Marketplace
  subscription; current account access was not checked. The harness can record
  a future COOL run, but no COOL performance or cost improvement is established.

## Detector micro-benchmark

x86_64, first **200** RDD2022-India images after one warm-up: median **229.55 ms**,
p95 **236.09 ms**, **4.34 images/s**. The Graviton run of this micro-benchmark is missing
(the image bundle shipped to the instance contained dangling symlinks, so 0 images loaded);
the per-stage `stage_detect_s` numbers above are the Graviton detector evidence.

## Cost evidence limits

The old cost table multiplied these two-CPU host timings by one-vCPU Fargate
rates. That does not establish Fargate cost per frame, so those estimates and
the ARM discount conclusion have been removed. A defensible comparison needs
matched task sizes, workload, builds, concurrency and current prices, plus
explicit treatment of storage, ALB, network, IPv4 and idle time.

`eval.bench_cool compare` omits costs by default. Explicit `--price-a` and
`--price-b` rates produce labeled hypothetical compute-only estimates; they
do not make these historical hosts comparable or establish a measured bill.
