# MargAI

MargAI is an OpenCV road-damage survey agent. It processes dashcam video,
detects RDD2022 cracks and potholes, tracks observations into instances,
preserves full-resolution evidence, and uses an approval-gated agent to draft
municipal work orders.

## Architecture

```text
dashcam + GPX
     |
     v
OpenCV 5 DNN -> quality/keyframes -> tracker/evidence -> result.json
                                      |
                                      v
                              agent tools + MockLLM/Bedrock
                                      |
                         work orders + human approval dashboard

AWS upload -> S3 uploads -> SQS -> Fargate worker -> S3 survey outputs
                                      |                 |
                                      +-> DynamoDB -----+-> ALB/FastAPI dashboard
```

## Local quickstart

```bash
python -m venv /home/ubuntu/venv
/home/ubuntu/venv/bin/pip install -e '.[api,dev]'
/home/ubuntu/venv/bin/python -m marg.vision.pipeline \
  --video /path/to/road-video.webm --out outputs/demo
/home/ubuntu/venv/bin/python -m marg.agent.loop \
  --result outputs/demo/result.json --llm mock --out outputs/demo/agent
/home/ubuntu/venv/bin/python -m marg.api --data outputs --port 8000
```

Open `http://localhost:8000`.

## AWS deployment

The deployment uses one public-IP Fargate task, an ALB, S3, SQS, DynamoDB,
ECR, and CloudWatch Logs. The default task is ARM64 Graviton-compatible,
1 vCPU and 2 GB memory.

```bash
aws configure
AWS_REGION=us-east-1 infra/deploy.sh
MARG_S3_BUCKET="$(aws cloudformation describe-stacks --stack-name margai \
  --query 'Stacks[0].Outputs[?OutputKey==`Bucket`].OutputValue' --output text)" \
  infra/seed_demo.sh
```

`infra/deploy.sh` discovers the default VPC and default subnets, builds and
pushes the image, deploys `infra/cloudformation.yaml`, and prints the ALB URL.
If ARM64 binfmt or the build fails, it retries with an X86_64 image and task
definition. The worker uses the MockLLM by default; set
`MARG_AGENT_LLM=bedrock` (and `MARG_BEDROCK_MODEL_ID`) to use Bedrock.

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `AWS_REGION` | AWS region | `us-east-1` |
| `MARG_S3_BUCKET` | Enables S3-backed API storage | unset |
| `MARG_S3_PREFIX` | S3 key prefix | empty |
| `MARG_S3_CACHE_DIR` | Local survey cache | `/tmp/marg-cache` |
| `MARG_SQS_QUEUE_URL` | Upload worker queue | unset |
| `MARG_AGENT_LLM` | Worker agent backend | `mock` |
| `MARG_BEDROCK_MODEL_ID` | Bedrock model ID | `us.amazon.nova-pro-v1:0` |
| `MARG_WORK_ORDERS_TABLE` | Work-order table | `margai-work-orders` |
| `MARG_SURVEYS_TABLE` | Survey status table | `margai-surveys` |

## Evaluation and tests

```bash
/home/ubuntu/venv/bin/ruff check .
/home/ubuntu/venv/bin/python -m pytest -q
/home/ubuntu/venv/bin/python eval/eval_agent.py --llm mock
```

## Licenses and attribution

- OpenCV is Apache-2.0.
- The exported RDD2022 YOLO model has an AGPL-3.0 source/export note; review
  the upstream terms before redistribution.
- RDD2022 is used under CC BY-SA 4.0.
- The pothole video fixtures are CC BY-SA 4.0 by Amuzujoe.
- The Bengaluru pothole image is Wikimedia Commons material; attribution is
  recorded in `tests/fixtures/LICENSES.md`.
