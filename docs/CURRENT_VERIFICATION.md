# Current source verification — 26 October 2026

The October 22 reviewed Modal source is now tracked alongside the evidence
workflow and October 26 detector latency measurement. SOURCE_PATCH_MANIFEST.json
records application hashes and the two client files changed for recorded-first
navigation. The original archive remains a dated snapshot, not today's Git tree.

On the existing Mac Python 3.12 environment, the integrated tree passed all
160 Python tests, 14 JavaScript tests, Ruff, JavaScript syntax and CloudFormation
lint. These offline checks do not establish new live inference or detector accuracy.
The primary recorded-review action was exercised in the local browser and opened
the existing eight-tool activity trace without a model request.

Known-pattern scanning found no credentials in staged text files. The scanner
omitted three binary images (the architecture diagram and two actual video frames);
it does not cover arbitrary credential formats or Git history.

The public demo runs on existing AWS Lightsail capacity, with Modal supplying
optional DeepSeek model calls. See JUDGE_DEPLOYMENT.md for reproduction boundaries,
JUDGING_GUIDE.md for the free recorded path, FRESH_LATENCY.md for today's timing
method, and VIDEO_VERIFICATION.md for the video evidence and team-credit limitation.
Historical live-review receipts remain in the October 22 judge archive. No new
paid inference was run for this source verification.

## Published and hosted checks

Source commit `332488ef39f04a652e876b5cef97d9dea02a8e3e` was fast-forwarded to
public master. [GitHub CI](https://github.com/himanshu748/marg-ai/actions/runs/40282304362)
passed on Python 3.10 with pinned dependencies and Node 22.

At 06:56 UTC, anonymous HTTP requests (no authentication or Cookie headers)
returned 200 for health, capabilities and all three client assets. Their hashes
match the committed files. The public browser's primary **View recorded review**
action opened the stored eight-tool trace with six of six policy checks; stored
work orders and linked evidence remained readable. No live-review button was run.

The first attempted Docker overlay build timed out before any service cutover.
Inspection established the old service was still running; no build remained
active. The final update reused the existing image with three read-only static
asset bind mounts, avoiding another build. Its previous container remains stopped
for rollback. Data and the allowance ledger were preserved, the ledger remained
31 requests / $0.318816 reserved, and protected existing services were unchanged.
No new cloud resource, purchase or inference request was created.

Host checks found restart-unless-stopped, persistent recorded-data and TLS volumes,
zero restarts before the update and 50.6 GB available disk. The certificate expires
20 January 2027. These are point-in-time checks; future host retention and uptime
remain operational dependencies. Existing repository evidence and video provide a
fallback through the judging window.

- [Anonymous HTTP and asset hashes](verification/anonymous-public-2026-10-26.json)
- [Deployment receipt](verification/deployment-2026-10-26.json)
- [Recorded-first screenshot](verification/recorded-first-2026-10-26.jpg)
- [Stored activity screenshot](verification/recorded-activity-2026-10-26.jpg)

The saved Devpost entry was read back as SUBMITTED, 5/5 complete after a description
update linking the published source and video caveat. The previously saved fresh
latency and Agentic Vision opt-in were preserved. No final submit or new agreement
was invoked. The saved YouTube link played at the results chapter during this
check; that browser was signed in, so this is not a new anonymous full-playback test.
