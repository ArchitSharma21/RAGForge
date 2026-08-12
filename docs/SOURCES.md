# Sources and implementation references - v2.0 final

## Final-release note

v2.0 introduces no new runtime dependency or external architectural component. The final changes are benchmark alignment, UI/documentation polish, and local release-verification tooling. Existing official/source references below continue to document the underlying Gradio, FastAPI, Gemini, LangGraph, Qdrant and retrieval behavior.

No tutorial source code is copied into RAGForge. The requested projects were used as architectural inspiration/checklists.

## Requested reference projects

1. Krish Naik — Enterprise Advanced RAG with Hybrid Search, Reranking, HyDE, CRAG, Self-RAG, Text2SQL, Caching and Guardrails in LangGraph  
   https://www.krishnaik.in/project/enterprise-advanced-rag-with-hybrid-search-reranking-hyde-crag-self-rag-text2sql-caching-and-guardrails-in-langgraph
2. Krish Naik — Production Grade Cyclic RAG with LangGraph, GCP and Groq  
   https://www.krishnaik.in/project/production-grade-cyclic-rag-with-langgraph-gcp-and-groq
3. Krish Naik — Building a RAG-Based Document Search Application  
   https://www.krishnaik.in/project/building-a-rag-based-document-search-application
4. Krish Naik — Air India RAG Chatbot Development  
   https://www.krishnaik.in/project/air-india-rag-chatbot-development
5. Educative — Building a Retrieval-Augmented Generation System Using FastAPI  
   https://www.educative.io/projects/building-a-retrieval-augmented-generation-system-using-fastapi
6. ByteByteAI — AI Engineering curriculum / Ask-the-Web modules  
   https://bytebyteai.com/c/ai-engineering

## Official implementation references

- Gemini Interactions API migration / structured output  
  https://ai.google.dev/gemini-api/docs/migrate-to-interactions
- Gemini Interactions structured-output schema update  
  https://ai.google.dev/gemini-api/docs/interactions-breaking-changes-may-2026
- LangGraph custom agentic RAG: document grading, conditional routing and query rewrite loop  
  https://docs.langchain.com/oss/python/langgraph/agentic-rag
- LangGraph Graph API / conditional edges  
  https://docs.langchain.com/oss/python/langgraph/graph-api
- Qdrant filtering/search relevance/hybrid retrieval  
  https://qdrant.tech/documentation/search/filtering/  
  https://qdrant.tech/documentation/search/search-relevance/  
  https://qdrant.tech/documentation/search/hybrid-queries/
- FastEmbed  
  https://github.com/qdrant/fastembed
- Hugging Face Spaces documentation  
  https://huggingface.co/docs/hub/spaces-overview
- Gradio BrowserState / state lifecycle
  https://www.gradio.app/guides/state-in-blocks
- Gradio Progress
  https://www.gradio.app/docs/gradio/progress

## Evaluation references

- RAGAS paper - component-wise/reference-free evaluation of RAG pipelines  
  https://arxiv.org/abs/2309.15217
- RAGAS context precision / context recall / faithfulness / response relevancy documentation  
  https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- LangSmith RAG evaluation tutorial - datasets plus answer/retrieval evaluators  
  https://docs.langchain.com/langsmith/evaluate-rag-tutorial
- Gradio event progress controls - `show_progress=hidden` suppresses the automatic overlay when explicit progress is used  
  https://www.gradio.app/docs/gradio/on
- Gemini API rate limits - active limits are project/model dependent and visible in Google AI Studio  
  https://ai.google.dev/gemini-api/docs/rate-limits
- Gemini troubleshooting - bounded exponential backoff for 429/5xx and retry guidance  
  https://ai.google.dev/gemini-api/docs/troubleshooting

## v1.5 evidence-driven runtime policy

The adaptive reranker policy and incremental evaluation design are internal engineering decisions informed by RAGForge's own transparent bundled benchmark. They are not presented as universal claims that reranking is ineffective or that one evaluation design is optimal for every corpus. The explicit reranker ablation remains available so the decision can be revisited when corpus size/difficulty changes.


## v1.6 evaluation-driven policy

Insight synthesis, hard-mode benchmark cases, table citation semantics, chunk-level reranker labels, profile comparison and evaluation-history deltas are RAGForge-specific engineering additions derived from the project's own observed evaluation gaps. They are not claims that one routing taxonomy or benchmark design is universally optimal.

## v1.7 evaluation-policy note

v1.7 does not add a new external dependency or benchmark dataset. The new policies are derived from RAGForge's own auditable v1.6 run data: perfect source/chunk ablation quality with large reranker latency, a false-negative missing-answer case, and Markdown citation-coverage artifacts.

## v1.8 evaluation-policy note

v1.8 adds no new external benchmark or hosted dependency. The context-budget policy is derived from RAGForge's own v1.7 auditable run: focused QA retained 100% source Recall@5/Hit@1/MRR while source Precision@5 remained roughly 47%, and generation dominated node latency. The new ablation therefore tests whether removing the focused-query distractor tail preserves recall before treating the optimization as beneficial.


## v1.9 implementation note

v1.9 adds no external runtime dependency and no new third-party architecture source. Adaptive budgets, sentence compression, scale stress, workspace diagnostics and readiness scoring are implemented locally on top of the existing retrieval/evaluation stack. The scale harness reuses existing embedding vectors rather than calling an embedding or generation API.