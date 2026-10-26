# Current AWS plus Modal deployment

Existing AWS Lightsail capacity in ap-south-1 hosts FastAPI, OpenCV tools, evidence, audit and human decisions. An existing Modal shared endpoint provides `deepseek-ai/DeepSeek-V4.1-Flash`. The CloudFormation/Fargate stack is an alternate implementation. These instructions require independently authorized existing capacity and credits. No credentials or SSH control scripts are bundled.

## Build and separate data

From the package root:

```sh
python fetch_models.py
docker build -f Dockerfile.judge -t margai-live:modal-20261022 .
cp -R demo judge-data
mkdir live-data
sudo chown -R 10001:10001 judge-data live-data
```

The deployed judge-data also contains the separately processed `aws-clip3s` survey. The earlier offline processing/privacy helper is preserved in the October 22 judge archive, not this checkout; its processing mount must be writable. Serving mounts recorded fixtures read-only. Live reviews clone only the two allowlisted samples into the separate live-data directory.

## Protected runtime variables

Create a mode-0600 runtime file outside source and Docker build context. Privately populate the inference proxy token; never put its value in command arguments, browser, repository or archive. Its combined format is Modal-Key, a literal period, then Modal-Secret. This Starter workspace token can authenticate to workspace inference endpoints and is not endpoint-scoped; it grants no administrative API access. No AWS root or Modal administrative credentials are needed.

```text
MARG_MODAL_ENABLED=1
MARG_MODAL_ENDPOINT=https://YOUR_EXISTING_ENDPOINT.modal.direct
MARG_MODAL_PROXY_TOKEN=<privately supplied inference proxy credential>
MARG_MODAL_ALLOWANCE_DB=/live/allowance.sqlite3
MARG_LIVE_DEMO_ENABLED=1
MARG_LIVE_DATA_ROOT=/live
MARG_DATA_ROOT=/data
MARG_READ_ONLY=1
MARG_BEDROCK_ENABLED=0
MARG_COOKIE_SECURE=1
MARG_HOSTING_LABEL=AWS Lightsail
MARG_JOB_WORKERS=1
MARG_JOB_CAPACITY=2
```

Production uses Authorization Bearer; do not set the local CLI testing override `MARG_MODAL_AUTH_HEADER`. Preserve the existing SQLite allowance ledger across restarts/rebuilds. The deployed ledger includes local verification reservations. Deleting it would reset the application limit and is not normal operation.

```sh
docker run -d --name margai-live --restart unless-stopped \
  --network YOUR_EXISTING_JUDGE_NETWORK --read-only --cap-drop ALL \
  --security-opt no-new-privileges --memory 768m --cpus 0.5 --pids-limit 128 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --mount type=bind,src="$PWD/judge-data",dst=/data,readonly \
  --mount type=bind,src="$PWD/live-data",dst=/live \
  --env-file /protected/path/runtime.env margai-live:modal-20261022
```

The existing Caddy route is:

```caddy
margai.13.204.212.172.sslip.io {
    encode gzip
    header X-Robots-Tag "noindex, nofollow"
    reverse_proxy margai-live:8080
}
```

Validate the complete proxy configuration before graceful reload, preserve other routes and retain the old container for rollback. Linux dependency installation and pip check were verified by the deployed image; the full suite ran on macOS ARM64. Base-image rebuilding and detector numerics are not claimed bit-for-bit reproducible.

## Verify the journey and limits

HTTPS `/api/health` retains `read_only: true` for recorded surveys and `bedrock_enabled: false`. `/api/capabilities` separately reports `live_reviews.enabled`, exact model and remaining allowance. Choose Open live review, then Run live review; inspect real provider receipts, OpenCV tools and passing audit, save a sample decision and reload. A separate session must receive 404 for the private job; the original recorded-agent write route remains 403. Public arbitrary prompts and uploads are disabled. Cookies are Secure, HttpOnly and SameSite Strict.

The persistent ledger reserves conservative cost before each request, including failures/timeouts. Hard limits are $0.50 lifetime reserved usage, 100 model requests, 1,024 output tokens/request, 64 KiB request payload, 20 review attempts globally, two/browser, one concurrent review and November 11, 2026 00:00 UTC expiry. No retry or mock fallback occurs on live failure. Existing additional grants fund usage; no purchase or top-up occurred. Provider prices/workspace settings can change externally. Application expiry stops requests but does not revoke the credential itself; retire it separately when no longer needed. Current receipts establish hosted inference, saved approval, session isolation and write protection.
