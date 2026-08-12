# Bundled demo corpus

The corpus mixes tiny original synthetic fixtures with one realistic public reference document:

- `acme_cloud_runbook.md` — synthetic incident-response runbook
- `orbitpay_policy.txt` — synthetic payments/dispute policy
- `support_matrix.csv` — synthetic structured data for RAG + Text2SQL
- `release_notes.html` — synthetic HTML release notes
- `NIST_AI_RMF_1.0.pdf` — NIST Artificial Intelligence Risk Management Framework 1.0 (January 2023), bundled as a realistic long-PDF test document; it retains its source/publication terms

Try:
- “What is the Sev-1 acknowledgement target?”
- “How long can a customer dispute a card transaction?”
- “Which support tier has the fastest first-response SLA?”
- “What are the four AI RMF functions?”
- In **Data (SQL)** mode: “What is the average monthly price across paid support tiers?”

See `docs/DEMO_DATASETS.md` for additional public/open corpora you can swap in.
