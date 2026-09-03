from typing import Any

from pydantic import BaseModel, Field


class RetrievedEvidence(BaseModel):
    """从Qdrant中召回的一条证据。"""

    evidence_id: int
    score: float
    rerank_score: float | None = None
    document_id: str

    chunk_id: str
    source_file: str
    section: str

    page_start: int | None = None
    page_end: int | None = None

    text: str
    source_locations: list[dict[str, Any]] = Field(
        default_factory=list
    )

class AnswerClaim(BaseModel):
    """模型生成的一条结论及其证据编号"""

    statement: str

    evidence_ids: list[int] = Field(min_length=1)


class GroundedAnswer(BaseModel):
    """由大模型生成的结构化回答"""

    sufficient_evidence: bool

    overview: str
    overview_evidence_ids: list[int]

    claims: list[AnswerClaim]
    limitations: list[str]


class RAGResult(BaseModel):
    """一次完整RAG问答的结果"""
    query: str
    document_ids: list[str] | None = None
    answer: GroundedAnswer
    evidences: list[RetrievedEvidence]
    retrieval_queries: list[str] = Field(default_factory=list)

