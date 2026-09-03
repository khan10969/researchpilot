import pytest

from researchpilot.rag.models import (
    AnswerClaim,
    GroundedAnswer,
    RetrievedEvidence,
)
from researchpilot.rag.service import RAGService


def make_evidence(
    evidence_id: int,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id,
        score=0.8,
        document_id="doc-1",
        chunk_id=f"chunk-{evidence_id}",
        source_file="test.pdf",
        section="实验结果",
        page_start=3,
        page_end=3,
        text="测试证据",
        source_locations=[],
    )


def test_valid_citations_pass() -> None:
    answer = GroundedAnswer(
        sufficient_evidence=True,
        overview="论文报告了实验结果。",
        overview_evidence_ids=[1],
        claims=[
            AnswerClaim(
                statement="识别率得到提升。",
                evidence_ids=[1],
            )
        ],
        limitations=[],
    )

    RAGService._validate_citations(
        answer=answer,
        evidences=[make_evidence(1)],
    )


def test_unknown_citation_is_rejected() -> None:
    answer = GroundedAnswer(
        sufficient_evidence=True,
        overview="论文报告了实验结果。",
        overview_evidence_ids=[99],
        claims=[],
        limitations=[],
    )

    with pytest.raises(
        RuntimeError,
        match="不存在的证据编号",
    ):
        RAGService._validate_citations(
            answer=answer,
            evidences=[make_evidence(1)],
        )
