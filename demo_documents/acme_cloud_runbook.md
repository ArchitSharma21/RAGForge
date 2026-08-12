# Acme Cloud Reliability Runbook

## Service objectives
The Checkout API has a monthly availability SLO of 99.95%. The latency SLO is that 95% of requests complete within 400 ms. Error-budget burn is reviewed every Monday.

## Incident severity
A Sev-1 incident is a complete outage, confirmed data-loss event, or payment-processing failure affecting more than 20% of traffic. The on-call engineer must acknowledge a Sev-1 page within **5 minutes** and establish an incident channel within 10 minutes.

A Sev-2 incident is a major degradation affecting at least 5% of requests. The acknowledgement target is 15 minutes.

## Safe rollback
For a suspected bad deployment, first freeze additional releases, compare the current and previous release health, and rollback only after confirming database migrations are backward-compatible. If rollback is unsafe, disable the affected feature flag and shift traffic to the healthy region.

## Recovery verification
Recovery is not complete when dashboards merely look green. Verify synthetic checkout, payment authorization, queue depth, and customer-facing error rates for at least 15 minutes before closing the incident.
