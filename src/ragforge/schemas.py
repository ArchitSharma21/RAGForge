from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from pydantic import BaseModel, Field


KnowledgeScope = Literal["corpus", "external", "mixed", "structured_data"]
TaskType = Literal[
    "fact_lookup",
    "overview",
    "cross_document_synthesis",
    "comparison",
    "aggregation",
    "insight_synthesis",
    "followup",
]
RetrievalStrategy = Literal["semantic", "global", "hierarchical", "analytical", "table", "none"]
WebRelevance = Literal["required", "useful", "irrelevant"]


@dataclass(slots=True)
class Document:
    text: str
    source: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    origin: Literal["document", "web"] = "document"
    url: str | None = None
    title: str | None = None


@dataclass(slots=True)
class SourceProfile:
    source: str
    file_type: str
    document_units: int
    chunk_count: int
    page_count: int
    section_count: int
    representative_chunk_ids: list[str]
    profile_text: str


class QueryPlan(BaseModel):
    """Semantic plan produced before retrieval.

    The fields deliberately separate *where knowledge lives* from *how it should
    be retrieved*. That keeps phrases such as "current corpus" from being
    mistaken for current-world/fresh-web intent.
    """

    route: Literal["documents", "web", "hybrid", "sql"] = "documents"
    knowledge_scope: KnowledgeScope = "corpus"
    task_type: TaskType = "fact_lookup"
    retrieval_strategy: RetrievalStrategy = "semantic"
    web_relevance: WebRelevance = "irrelevant"
    requires_fresh_web: bool = False
    rewritten_query: str = ""
    document_queries: list[str] = Field(default_factory=list)
    web_queries: list[str] = Field(default_factory=list)
    hyde: str = ""
    rationale: str = ""


class EvidenceAssessment(BaseModel):
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    top_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    method_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    source_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    unique_sources: int = 0
    corpus_sources: int = 0
    sufficient: bool = False
    reason: str = ""


class RAGEvalJudgement(BaseModel):
    """Auxiliary LLM-as-judge scores for benchmark cases.

    These scores complement deterministic labels; they are never treated as
    ground truth because judge models can be noisy or biased.
    """

    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_support: float = Field(default=0.0, ge=0.0, le=1.0)
    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    pass_: bool = Field(default=False, alias="pass")
    reason: str = ""

    model_config = {"populate_by_name": True}


class PipelineConfig(BaseModel):
    mode: Literal["Auto", "Documents", "Web", "Hybrid", "Data (SQL)"] = "Auto"
    profile: Literal["Fast", "Balanced", "Agentic"] = "Balanced"
    model: str = "gemini-3.5-flash-lite"
    web_provider: Literal["Auto", "DuckDuckGo", "Tavily", "Gemini Search"] = "Auto"
    use_hyde: bool = True
    use_multi_query: bool = True
    use_reranker: bool = True
    use_context_pruning: bool = True
    use_adaptive_top_k: bool = True
    use_evidence_compression: bool = True
    use_crag: bool = True
    use_self_rag: bool = True
    allow_web_fallback: bool = True
    use_history: bool = True
    top_k: int = Field(default=6, ge=2, le=12)


class QueryRequest(BaseModel):
    session_id: str
    query: str = Field(min_length=1, max_length=8000)
    config: PipelineConfig = Field(default_factory=PipelineConfig)


class EvaluationRequest(BaseModel):
    session_id: str
    level: Literal["Quick", "Standard", "Deep"] = "Standard"
    model: str = "gemini-3.5-flash-lite"
    target_rpm: int = Field(default=12, ge=0, le=60)
    reuse_saved: bool = True
    include_profile_benchmark: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    trace: dict[str, Any]
    confidence: float


class SessionResponse(BaseModel):
    session_id: str


class CorpusSummary(BaseModel):
    session_id: str
    documents: int
    chunks: int
    tables: list[str]
    sources: list[str]
    source_profiles: int = 0
