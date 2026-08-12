# Migration to v1.4.1

v1.4.1 is a stabilization patch over v1.4. It does not change the core retrieval architecture or demo corpus.

## Fixes

- fixes `Architecture + API -> Refresh runtime view`, which could raise `NameError: name 'f' is not defined` because the generated curl-example string was truncated in the v1.4 packaged UI source;
- changes visible evaluation wording from `scorecard` to `score card`;
- adds quota-safe Gemini evaluation pacing, defaulting to 12 RPM;
- records recent interactive requests in the same process-local per-key/per-model request ledger used by evaluation pacing;
- honors surfaced Gemini 429 retry guidance before bounded retries;
- prevents structured-output fallback code from immediately issuing another API request after transient 429/5xx failures;
- reports deliberate pacing wait separately from service/pipeline latency;
- reduces Text2SQL evaluation from roughly three Gemini calls per case to one by separating planner-routing evaluation from SQL generation/execution evaluation;
- reduces Deep judge calls to a representative labeled benchmark subset.

## Recommended free-tier setting

Use the active RPM displayed for your API project in Google AI Studio as the source of truth. The UI defaults to 12 RPM, which leaves headroom when a project currently has a 15 RPM limit. Lower the target if the same API key is also receiving traffic outside this Space.

## Upgrade

Apply the v1.4.1 patch over a clean v1.4 tree, then commit normally. The patch does not contain the bundled NIST PDF.

After deployment:

1. run Quick with quota-safe pacing enabled;
2. run Standard and confirm request/pacing telemetry appears in the score card;
3. run Deep immediately afterward to test rolling-window continuity;
4. open `Architecture + API` and click `Refresh runtime view`;
5. confirm the runtime JSON and curl examples populate without a traceback.
