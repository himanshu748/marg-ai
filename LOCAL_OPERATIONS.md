# Work locally while Bedrock access is pending

The deterministic mock agent exercises survey review, tool traces, draft work
orders, and human decisions without AWS credentials or Bedrock calls. It does
not demonstrate live model reasoning. Keep this distinction in demos and reports.

Start with the `.venv` setup and copied demo data in [README.md](README.md#local-quickstart).
Serve `outputs/workbench`, not the tracked `demo/` directory. In the dashboard,
upload a video or select a survey, run deterministic review, inspect the evidence
and trace, then approve or reject a pending work order.

## Access and configuration

Keep writable development bound to `127.0.0.1`. Set `MARG_READ_ONLY=1` for a
view-only workspace; this blocks uploads, agent runs, and decisions. Without an
API token, reads are public to anyone who can reach the server.

`MARG_API_TOKEN` protects survey APIs, media, jobs, and writes. The dashboard
exchanges the token for a one-hour HttpOnly, SameSite=Strict session cookie.
Use `MARG_COOKIE_SECURE=1` behind HTTPS. Static assets, health, capabilities, and
the session status endpoint remain public. Do not put token values in source,
URLs, frontend code, or CloudFormation parameters.

| Variable | Meaning | Default |
| --- | --- | --- |
| `MARG_DATA_ROOT` | Local surveys and job ledger | `outputs` |
| `MARG_READ_ONLY` | Set `1` to disable data mutations | unset for local development |
| `MARG_API_TOKEN` | Shared workspace access token | unset |
| `MARG_COOKIE_SECURE` | Set `1` for HTTPS-only cookies | unset |
| `MARG_MAX_VIDEO_BYTES` | Maximum individual video size | 268435456 (256 MiB) |
| `MARG_MAX_GPX_BYTES` | Maximum GPX size | 5242880 (5 MiB) |
| `MARG_JOB_WORKERS` | Local processing threads | 2 |
| `MARG_JOB_CAPACITY` | Local running and queued job limit | 8 |
| `MARG_AGENT_LLM` | Worker provider; set `mock` explicitly for local work | `mock` unless a model ID selects Bedrock |
| `MARG_BEDROCK_ENABLED` | Set `1` to permit Bedrock invocation | disabled |
| `MARG_BEDROCK_MODEL_ID` | Bedrock model or inference profile | `us.amazon.nova-pro-v1:0` |
| `MARG_ALLOW_INSECURE_WRITES` | Set `true` only for token-authenticated HTTP judging deployments | `false` |
| `AWS_REGION` | Bedrock region | `us-east-1` |
| `MARG_S3_BUCKET` | Enable AWS survey storage | unset |
| `MARG_S3_PREFIX` | Survey/upload object prefix | empty |
| `MARG_S3_CACHE_DIR` | API cache and local job ledger with S3 enabled | `MARG_DATA_ROOT` when omitted by API |
| `MARG_SQS_QUEUE_URL` | AWS upload processing queue | unset |
| `MARG_WORK_ORDERS_TABLE` | DynamoDB work orders | `margai-work-orders` |
| `MARG_SURVEYS_TABLE` | DynamoDB survey status | `margai-surveys` |

The API and local upload path select `mock` by default, independent of the
worker provider. Enabling Bedrock exposes the explicit `llm=bedrock` API option;
it does not prove entitlement or model availability and never falls back to mock.

## Jobs and persistence

Local uploads accept MP4, MOV, AVI, MKV, or WebM plus optional GPX with timestamped
track points. Unreadable video, empty files, invalid GPX, and oversized files are
rejected. Without GPX, survey coordinates are synthetic and must not be presented
as recorded GPS. Local uploads run vision and deterministic review. If vision
succeeds but review fails, evidence stays available for a review retry.

Jobs progress through `queued`, `running`, `succeeded`, or `failed`. The ledger
is `<data-root>/.marg/jobs.sqlite3`; with AWS storage it lives under the cache
root. Preserve this directory on a retained volume for job history to survive
container replacement. Interrupted local jobs become failed on restart and are
not replayed automatically. Local uploaded originals are deleted after processing
or interruption cleanup; `evidence_raw/` remains private inspection material.

Run one API process and one replica. Session state is process-local; the journal
and filesystem locks are not a distributed job coordinator. SQS jobs do not
consume the local executor limit: SQS quotas, worker concurrency, visibility,
and the dead-letter queue govern that path. Concurrent SQS redelivery still
needs a distributed execution lock before adding workers or replicas.

Duplicate local survey/agent jobs return HTTP 409 with the existing job URL. A
full local queue returns 429 and `Retry-After`. Decisions require a completed,
passing review and a pending work order. The same decision is idempotent; an
opposite final decision or stale cloud run returns 409. Run IDs prevent an older
human approval from being applied to a newer review.

## API contract

Responses use JSON; upload is multipart. With token authentication enabled,
clients can send `Authorization: Bearer <token>` or use the session cookie.

| Method and path | Result |
| --- | --- |
| `GET /api/health` or `/healthz` | Service status and read-only/storage configuration |
| `GET /api/capabilities` | Upload limits, privacy readiness, auth requirement, provider configuration |
| `POST /api/session` | Exchange bearer token for the HttpOnly cookie |
| `GET /api/session`, `DELETE /api/session` | Session status and logout |
| `GET /api/surveys` | Completed, processing, and failed surveys; latest agent job |
| `GET /api/surveys/{id}` | Vision result |
| `GET /api/surveys/{id}/agent` or `/trace` | Current review, provenance, audit, and tool evidence |
| `POST /api/surveys/upload` | `video`, optional `gpx`, optional `name`; 202 with a job |
| `POST /api/surveys/{id}/run_agent?llm=mock` | 200 with a queued review job |
| `GET /api/jobs/{job_id}` | Job state, timestamps, error, and result link when available |
| `GET /api/surveys/{id}/status` | Latest job with separate survey/agent job records |
| `POST /api/surveys/{id}/work_orders/{order_id}/decision` | JSON `decision: approve|reject`, optional `note` up to 2000 characters |

Upload and run responses include `job_id`, `survey_id`, `status`, and `status_url`.
Poll `status_url` until terminal state; an HTTP 200 from a status endpoint does
not itself mean processing succeeded. Legacy seeded surveys retain `status: done`
in the survey listing. Provider failures appear in the agent result and job state.

Cloud reads refresh result, review, and decision metadata. A cloud review hydrates
inspection images before running and publishes resulting inspection crops. This
avoids decisions depending on which images a browser previously cached.

## Privacy checks

Face and plate ONNX models must both load before a redacted pipeline run begins.
If either detector fails during processing, the run fails rather than publishing
unchecked images. Sensitive detections always take precedence over overlapping
road-damage boxes, including inspection crops. Keyframe detection uses the
original image before damage annotations are drawn.

Automated detectors can still miss faces and plates. Review exported imagery
before public sharing. `evidence_raw/` is private re-inspection material and must
never be exposed through a static file server or included in public downloads.
Previously generated artifacts are not retroactively fixed; rerun the survey
before publishing them. Disabling `VisionConfig.redact` is only suitable for
private experimentation.

## Future AWS deployment

No deployment is needed for local UI or backend development. The deployment
script creates billable resources when run; it has not been run as part of these
changes. The template now defaults to `ReadOnly=true` and `AgentLlm=mock`.

A writable deployment needs all of the following:

- An ACM certificate in the deployment region covering your dashboard domain.
- DNS for that domain pointing to the `LoadBalancerDNS` stack output.
- An AWS Secrets Manager secret containing the API token as its entire plaintext
  value, using the default Secrets Manager encryption key. A custom KMS key needs
  an additional scoped `kms:Decrypt` permission on the task execution role.
- `MARG_DEPLOY_READ_ONLY=false`, `MARG_CERTIFICATE_ARN`,
  `MARG_DASHBOARD_DOMAIN`, and `MARG_API_TOKEN_SECRET_ARN` in the deployment
  environment. The last value is the secret ARN, never the token itself.

With a certificate, the ALB serves HTTPS using TLS 1.2/1.3 and redirects HTTP.
Use the `DashboardURL` output (the certificate does not cover the raw ALB DNS
name). Secure session cookies are enabled on HTTPS deployments. Without a
certificate, the deployment stays read-only unless token-authenticated
insecure writes are explicitly enabled for judging. Only publish sanitized
demo datasets on an anonymous read-only deployment.

The deployment script uses `cfn-lint` when it is available on `PATH` or under
`.venv/bin`; otherwise it warns and skips template validation. Install it with
the `dev` extra for local validation. The script validates security prerequisites
before calling AWS, builds an ARM64 image by default, and uses a new timestamped
image tag. It does not silently switch a Graviton build to x86 or install
privileged emulation helpers. Configure your Docker builder for the chosen
platform; choose `MARG_CPU_ARCHITECTURE=X86_64` explicitly if that is the
intended deployment.

Writable judging deployment without an ACM certificate requires a Secrets
Manager token and explicit insecure-write opt-in:

```bash
aws secretsmanager create-secret --name margai/api-token --secret-string "$(openssl rand -hex 24)"
MARG_DEPLOY_READ_ONLY=false MARG_ALLOW_INSECURE_WRITES=true \
MARG_API_TOKEN_SECRET_ARN=<arn> AWS_REGION=us-east-1 infra/deploy.sh
```

The token travels in plaintext over HTTP; use this only during the judging
window.

Existing table, cluster, and log names remain unchanged to avoid replacing live
resources during a later stack update. This template therefore remains a
single-stack deployment until those names are migrated deliberately. S3 data is
encrypted and retained on stack deletion; object versions expire after 30 days.
DynamoDB tables are also retained on deletion. Retention may continue to incur
storage charges after compute resources are deleted.

Once a real Bedrock request succeeds in this account and region, set
`MARG_BEDROCK_ENABLED=1` and `MARG_AGENT_LLM=bedrock` with the intended
`MARG_BEDROCK_MODEL_ID`, then validate
one complete survey run. Marketplace agreement state and Nova/Claude access are
separate from local readiness.

## Verification

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/cfn-lint infra/cloudformation.yaml
bash -n infra/deploy.sh
```

These checks verify local behavior and template schema. They do not verify an AWS
deployment, the DNS/certificate pairing, provider access, or live account quotas.
