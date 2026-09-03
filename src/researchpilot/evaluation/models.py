"""Pydantic models used by the deterministic evaluation pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

EndpointName = Literal["qa", "experiment_analysis"]
SuiteName = Literal["qa", "experiment", "all"]


class GoldPoint(BaseModel):
    """An atomic fact that a high-quality answer should cover."""

    point_id: str
    description: str


class GoldEvidence(BaseModel):
    """Stable evidence location; runtime evidence IDs are deliberately excluded."""

    point_ids: list[str] = Field(default_factory=list)
    document_id: str
    acceptable_pages: list[int] = Field(default_factory=list)
    section: str | None = None
    anchor_terms: list[str] = Field(default_factory=list)


class AllowedContext(BaseModel):
    """Optional nearby evidence allowed for an unanswerable question."""

    description: str
    document_id: str
    acceptable_pages: list[int] = Field(default_factory=list)
    anchor_terms: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    """One line in an evaluation JSONL file."""

    case_id: str
    endpoint: EndpointName
    scope: Literal["single_document", "multi_document"]
    question_type: str
    difficulty: Literal["easy", "medium", "hard"]

    question: str | None = None
    focus: str | None = None
    document_ids: list[str] = Field(default_factory=list)

    answerability: Literal["full", "partial", "none"]
    expected_sufficient_evidence: bool
    expected_document_coverage: list[str] = Field(default_factory=list)

    gold_points: list[GoldPoint] = Field(default_factory=list)
    gold_evidence: list[GoldEvidence] = Field(default_factory=list)
    expected_limitations: list[str] = Field(default_factory=list)
    allowed_context: list[AllowedContext] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)

    required_summary_fields: list[str] = Field(default_factory=list)
    required_comparison_aspects: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_endpoint_input(self) -> "EvalCase":
        if self.endpoint == "qa" and not (self.question or "").strip():
            raise ValueError("qa 案例必须提供非空 question")

        if self.endpoint == "experiment_analysis" and not (self.focus or "").strip():
            raise ValueError("experiment_analysis 案例必须提供非空 focus")

        return self


class CaseMetrics(BaseModel):
    """Deterministic metrics for one evaluated case."""

    request_ok: bool
    actual_sufficient_evidence: bool | None = None
    sufficient_evidence_correct: bool | None = None

    citation_validity: float | None = None
    citation_presence_ok: bool | None = None
    cited_evidence_ids: list[int] = Field(default_factory=list)
    invalid_citation_ids: list[int] = Field(default_factory=list)

    document_coverage: float | None = None
    missing_document_ids: list[str] = Field(default_factory=list)

    gold_page_recall: float | None = None
    anchor_evidence_recall: float | None = None
    missed_gold_evidence_indexes: list[int] = Field(default_factory=list)

    structure_completeness: float | None = None
    missing_structure_items: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    """Stored result for one API request."""

    case_id: str
    endpoint: EndpointName
    request_body: dict[str, Any]
    status_code: int | None = None
    latency_ms: float
    response: dict[str, Any] | None = None
    error: str | None = None
    metrics: CaseMetrics
    passed: bool


class RunSummary(BaseModel):
    """Machine-readable aggregate for one complete evaluation run."""

    run_id: str
    created_at: str
    base_url: str
    qa_top_k: int
    analysis_top_k_per_document: int

    case_count: int
    request_success_count: int
    passed_count: int
    failed_count: int

    request_success_rate: float | None = None
    pass_rate: float | None = None
    sufficiency_accuracy: float | None = None
    citation_validity: float | None = None
    document_coverage: float | None = None
    gold_page_recall: float | None = None
    anchor_evidence_recall: float | None = None
    structure_completeness: float | None = None

    mean_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
