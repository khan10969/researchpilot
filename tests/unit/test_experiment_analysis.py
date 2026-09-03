import json

import pytest

from researchpilot.analysis.experiment_service import (
    ExperimentAnalysisService,
)
from researchpilot.analysis.models import (
    AnalysisEvidence,
    AnalysisLimitation,
    DocumentExperimentSummary,
    EvidenceBackedClaim,
    ExperimentAnalysis,
)


def empty_summary(
    document_id: str = "doc-a",
) -> DocumentExperimentSummary:
    return DocumentExperimentSummary(
        document_id=document_id,
        source_file="paper-a.pdf",
        research_tasks=[],
        methods=[],
        datasets=[],
        baselines=[],
        experiment_settings=[],
        metrics=[],
        results=[],
        limitations=[],
    )


def test_markdown_wrapped_json_is_accepted() -> None:
    payload = {
        "sufficient_evidence": False,
        "document_summaries": [],
        "comparisons": [],
        "overall_conclusion": "证据不足",
        "overall_evidence_ids": [],
        "limitations": [
            "没有找到实验结果。"
        ],
    }

    raw_output = (
        "```json\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
        )
        + "\n```"
    )

    cleaned = (
        ExperimentAnalysisService
        ._extract_json_object(raw_output)
    )

    parsed = (
        ExperimentAnalysis
        .model_validate_json(cleaned)
    )

    assert parsed.sufficient_evidence is False
    assert parsed.overall_conclusion == "证据不足"


def test_limitation_may_have_no_evidence() -> None:
    summary = empty_summary()

    summary.limitations.append(
        AnalysisLimitation(
            statement="当前证据未报告训练参数。",
            evidence_ids=[],
        )
    )

    assert (
        summary.limitations[0].evidence_ids
        == []
    )


def test_summary_cannot_cite_other_document() -> None:
    evidence = AnalysisEvidence(
        evidence_id=2,
        evidence_type="text_chunk",
        document_id="doc-b",
        source_file="paper-b.pdf",
        section="实验结果",
        page_start=4,
        page_end=4,
        retrieval_score=0.8,
        text="另一篇论文的证据",
    )

    summary = empty_summary(
        document_id="doc-a"
    )

    summary.results.append(
        EvidenceBackedClaim(
            statement="错误引用了另一篇论文。",
            evidence_ids=[2],
        )
    )

    analysis = ExperimentAnalysis(
        sufficient_evidence=True,
        document_summaries=[summary],
        comparisons=[],
        overall_conclusion="",
        overall_evidence_ids=[],
        limitations=[],
    )

    # 跳过 __init__，避免加载模型和连接外部服务。
    service = object.__new__(
        ExperimentAnalysisService
    )

    with pytest.raises(
        RuntimeError,
        match="其他文献的证据",
    ):
        service._validate_analysis(
            analysis=analysis,
            evidences=[evidence],
            documents={
                "doc-a": "paper-a.pdf"
            },
        )