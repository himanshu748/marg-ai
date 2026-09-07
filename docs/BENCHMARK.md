# OpenCV COOL benchmark

This report records the x86/pip-OpenCV baseline for the two demo survey videos. The benchmark harness is reproducible with `python -m eval.bench_cool run`; it performs one warm-up and two timed pipeline runs per video.

## x86 baseline

| Video | Label | Architecture | Source FPS | Processed FPS | Median wall (s) | Detector median (ms) | Keyframes | Cost / 1,000 source frames |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cars_moving_into_pothole_CC_BY_SA_4.0.webm | x86_pip_opencv | x86_64 | 7.70 | 2.57 | 80.65 | 229.55 | 14 | $0.001781 |
| pothole_kumasi_CC_BY_SA_4.0.ogv | x86_pip_opencv | x86_64 | 7.32 | 2.44 | 57.42 | 229.55 | 15 | $0.001875 |

## Detector micro-benchmark

The first **200** images from the RDD2022 India asset were measured after one detector warm-up: median **229.55 ms**, p95 **236.09 ms**, and **4.34 images/s**.

## Environment

- CPU: `Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz`; machine: `x86_64`; logical CPUs: `2`
- OpenCV: `5.0.0` from `opencv-python`; OpenCV threads: `2`
- Parallel framework: `pthreads`
- CPU/HW baseline: `SSE SSE2 SSE3`; dispatched: `not reported`
- COOL build marker present: `False`

The cost estimate uses the us-east-1 Linux Fargate list rates for 1 vCPU + 2 GB: x86 `$0.04048 + 2 × $0.004445 = $0.04937/hour` and ARM `$0.03238 + 2 × $0.00356 = $0.03950/hour`. These rates were checked against the [AWS Fargate pricing page](https://aws.amazon.com/fargate/pricing/). Storage, data transfer, and public IPv4 charges are excluded.

Graviton + COOL: pending AWS account activation; run `eval/bench_cool.py run --label graviton_cool` inside the arm64 container.
