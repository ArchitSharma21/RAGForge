# Suggested real-world demo files

The repository bundles several tiny synthetic fixtures plus the NIST AI Risk Management Framework 1.0 PDF, giving the one-click demo both fast deterministic checks and a realistic long-document workload. If you add more data, prefer one or two public/open documents rather than a huge corpus.

## Strong choices

1. **NIST AI Risk Management Framework 1.0 (PDF)** — excellent for policy/AI questions; U.S. government publication. Official page: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10
2. **NIST Generative AI Profile (PDF)** — useful for safety/evaluation questions. Official page: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence
3. **Kubernetes documentation** — Apache-2.0 licensed docs; useful to demonstrate technical troubleshooting and exact keyword retrieval. https://kubernetes.io/docs/
4. **NASA technical reports** — public U.S. government material; good for longer scientific PDFs. https://ntrs.nasa.gov/
5. **Project Gutenberg public-domain books** — useful for long-document retrieval and chapter citations. https://www.gutenberg.org/
6. **SEC EDGAR filings** — public-company filings, useful for tables and financial-document RAG. https://www.sec.gov/edgar

## Portfolio tip

Keep the one-click demo corpus small enough to index in seconds on CPU Basic. Put large optional PDFs in the repository only if the cold-start and indexing experience remains acceptable.

## v1.6 evaluation use

The five bundled sources remain the public demo corpus. v1.6 does not add hidden benchmark documents to the user corpus. Hard Mode instead stresses the same indexed sources with paraphrases, distractors, missing-answer questions, multi-hop comparisons, analytical synthesis and structured filtering; prompt-injection detection is tested as a security component case.
