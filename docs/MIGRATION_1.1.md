# RAGForge 1.1 migration

This release upgrades routing/retrieval policy without changing the public REST request schema.

## Main behavioral changes

- Balanced/Agentic Auto mode now uses a structured semantic `QueryPlan` rather than freshness keyword routing.
- The planner receives a compact session corpus manifest and recent history.
- Every corpus has a second source-profile index for source-level retrieval.
- Retrieval strategies are task-aware: semantic, hierarchical, global/source-balanced, table, or none.
- Document and web search queries are planned independently.
- CRAG performs one corrective document retrieval attempt before considering web fallback.
- Web fallback requires semantic relevance in addition to the UI permission flag.
- Overview/cross-document evidence grading includes source coverage.
- Source UI no longer presents raw cross-encoder logits as human relevance scores.
- Evaluation now measures semantic route/task accuracy and web-use precision/recall in addition to QA retrieval.

## Deployment fixes retained

- `requirements.txt` pins the known-compatible Gradio 5 / Google GenAI / Pydantic / FastAPI set.
- Docker runtime/cache directories are owned by the non-root `user` account.
- `.gitattributes` tracks PDFs through the Hugging Face Xet/LFS filter.

## Upgrade an existing Space

Copy the replacement files over the repository, then:

```bash
git add .
git commit -m "Upgrade RAGForge semantic routing and corrective retrieval"
git push origin main
```

After the Space rebuilds, click **Reset session**, re-index the demo corpus, and test:

```text
What is the corpus about?
```

In the Pipeline inspector you should see a plan similar to:

```text
route: documents
knowledge_scope: corpus
task_type: overview
retrieval_strategy: global
web_relevance: irrelevant
```

The exact planner wording may vary by model, but the answer should use document citations and should not run the web node for this session-local overview.
