# Query planning - v2.0 final

## Final planner contract

The recommended general-purpose profile remains `Auto + Balanced`. v2.0 also aligns the multi-hop Hard Mode reference case with that semantic path so multi-source comparison is evaluated as `comparison -> hierarchical` rather than through Fast mode's intentionally cheap fallback. No phrase-specific production route was added.

## v1.9 task/strategy consistency and scale-aware retrieval

The semantic planner remains schema-constrained, but v1.9 applies structural normalization after parsing so equivalent task/strategy pairs do not drift: `overview` uses `global`, `insight_synthesis` uses `analytical`, and comparison/cross-document work cannot remain a simple semantic-only strategy. This is task-level normalization rather than phrase-specific routing.

Retrieval depth is independently scale-aware. The planner decides *what kind* of retrieval is required; the runtime decides *how many candidates* are needed for the current corpus and then chooses a smaller generation context only when the task is focused and retrieval confidence permits it.

## 1. QueryPlan

Balanced and Agentic requests begin with a schema-constrained semantic analysis step. The plan contains:

- `route`: documents / web / hybrid / sql
- `knowledge_scope`: corpus / external / mixed / structured_data
- `task_type`: fact_lookup / overview / cross_document_synthesis / comparison / aggregation / insight_synthesis / followup
- `retrieval_strategy`: semantic / global / hierarchical / analytical / table / none
- `web_relevance`: required / useful / irrelevant
- `requires_fresh_web`
- standalone `rewritten_query`
- independent `document_queries`
- independent `web_queries`
- optional HyDE passage

The planner receives a compact manifest of the currently indexed corpus. Therefore temporal adjectives are interpreted in context: “current corpus” is local session state; “current exchange rate” is external freshness.

## 2. Source profiles

At ingestion, RAGForge groups document units/chunks by source and builds a deterministic profile from:

- source name/type
- document-unit, page, section and chunk counts
- representative excerpts sampled across the source

Profiles are embedded into a second retrieval-only index. They do not become answer evidence or synthetic citations.

## 3. Retrieval strategies

### semantic
Normal dense + BM25 + RRF + optional cross-encoder reranking across chunks.

### hierarchical
1. Retrieve source profiles.
2. Select relevant sources.
3. Search chunks only inside selected sources.
4. Diversify across sources for comparison/cross-document tasks.

### global
Choose source-balanced evidence so broad corpus summaries are not dominated by a long document. When the corpus contains more sources than final `top_k`, the source-profile index selects the most relevant subset.


### analytical
Used for `insight_synthesis` questions that ask what the indexed collection reveals, which patterns/trends stand out, or what important takeaways emerge across local evidence. The pipeline combines:

1. source-balanced original document chunks;
2. deterministic DuckDB table schema, bounded rows and descriptive signals;
3. one grounded synthesis step that distinguishes observations from interpretation.

Structured evidence is exposed as `[T#]` citations. This path is intentionally different from `table`: `table` computes a specific structured answer, while `analytical` synthesizes patterns across documents and tables.

### table
Use the isolated DuckDB/Text2SQL path.

### none
Used when documents are not part of the information need.

## 4. Corrective RAG policy

The correction loop is intentionally different from “weak score → web”.

```text
retrieve
  ↓
task-aware evidence grade
  ├─ sufficient → generate
  └─ weak
       ↓
     correct query/strategy
       ↓
     retrieve again
       ↓
     re-grade
       ├─ web semantically relevant + allowed → web
       └─ web irrelevant → explicit abstain
```

The first correction can rewrite document queries and change a focused semantic strategy to hierarchical/global retrieval. It cannot convert a private corpus-only information need into an external-only task simply because retrieval was weak.


## 4.5 Workspace preflight

RAGForge v1.4 treats missing local state as a lifecycle condition rather than a retrieval score. After semantic routing:

- a document route with zero indexed chunks terminates in `abstain`;
- a table/SQL route with zero tables terminates in `abstain`;
- a hybrid route with no local corpus may preserve the external half only when web information is semantically relevant and permitted.

This prevents an empty workspace from flowing through retrieval, generation, verification and revision as if it were merely a difficult question. The public UI can separately rebuild the bundled demo corpus before the graph runs.

## 5. Evidence grading

The local evidence score uses calibrated-ish retrieval signals rather than RRF or reranker raw values:

- top dense/BM25 relevance
- mean top-3 relevance
- dense/sparse method agreement
- distinct-source coverage

Weights vary by task:

- focused facts prioritize relevance
- comparisons/cross-document synthesis increase source diversity weight
- overview/global tasks strongly prioritize source coverage

Borderline Balanced cases and Agentic cases may also use a semantic LLM evidence judge.

## 6. Web permission vs relevance

`allow_web_fallback=True` means the application *may* use web if the plan says external information is relevant. It does not force web on a low retrieval score. Explicit Web/Hybrid route selection remains an override.

## 7. Regression tests

The evaluation harness includes session-local ambiguities such as:

- “What is the corpus about?”
- “I meant the current corpus that we have — what is that about?”

The expected behavior is document routing, overview/global retrieval, broad source coverage, and no web usage. These are behavioral tests only; no application rule matches those literal phrases.

## v1.5 adaptive reranking policy

The semantic planner still decides *what* retrieval strategy is needed. A separate runtime policy decides whether the local cross-encoder is worth its CPU latency for that plan.

When the reranker switch is enabled:

- Fast profile skips the cross-encoder;
- global/source-profile overview retrieval skips it because source balancing already determines corpus breadth;
- Balanced small-corpus focused lookups can skip it when the demo ablation shows no source-ranking benefit;
- comparison/cross-document tasks can retain it;
- larger corpora and Agentic profile can retain it.

The `retrieve`/`web` trace records `reranker_used` and `reranker_reason`. The explicit Standard/Deep ablation still runs both Hybrid RRF and Hybrid + reranker so the policy remains measurable rather than assumed.

## v1.6 insight-routing rules

- “What files are here?” remains `overview -> global`.
- “What exactly does this collection reveal? What trends stand out?” becomes `insight_synthesis -> analytical`.
- “Which tier has the shortest SLA?” remains `aggregation -> table`.
- “Compare our NIST document with the latest online guidance” remains `comparison -> hierarchical` with mixed/web relevance.

The planner is explicitly told that a collection-wide insight question should stay local/analytical even when structured tables are present; the existence of a table alone does not force SQL routing.

## v1.7 grounded absence behavior

A corpus-scoped fact lookup can legitimately conclude that the indexed evidence does not state the requested fact. v1.7 treats an evidence-cited absence statement as calibrated uncertainty, not as a hallucination signal. This does not change routing to the web: web fallback still requires semantic relevance/permission. The grounded-absence state simply prevents an unnecessary answer-revision call when the model has already answered conservatively from the local evidence.

## v1.8 focused context-budget behavior

Query planning still decides task/scope/strategy before any pruning. Context budgeting is deliberately a post-retrieval optimization, not a new routing heuristic.

Eligible runtime path:

```text
corpus/hybrid local evidence
+ fact_lookup or followup
+ semantic or hierarchical retrieval
+ context pruning enabled
-> keep top 3 ranked chunks
```

Ineligible tasks keep full breadth:

```text
overview
insight_synthesis
comparison
cross_document_synthesis
analytical/global retrieval
```

This separation prevents a latency optimization from silently redefining the user's information need. For focused fact lookups, generation also omits the full corpus manifest because the planner has already established corpus scope; broad and mixed tasks retain it.