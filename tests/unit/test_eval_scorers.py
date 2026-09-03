from researchpilot.evaluation.models import EvalCase
from researchpilot.evaluation.scorers import case_passed, score_response


def _case() -> EvalCase:
    return EvalCase.model_validate(
        {
            "case_id": "qa_cross_001",
            "endpoint": "qa",
            "scope": "multi_document",
            "question_type": "comparison",
            "difficulty": "medium",
            "question": "比较两篇论文。",
            "document_ids": ["doc-1", "doc-2"],
            "answerability": "full",
            "expected_sufficient_evidence": True,
            "expected_document_coverage": ["doc-1", "doc-2"],
            "gold_evidence": [
                {
                    "point_ids": ["p1"],
                    "document_id": "doc-1",
                    "acceptable_pages": [3],
                    "anchor_terms": ["注意力 机制"],
                },
                {
                    "point_ids": ["p2"],
                    "document_id": "doc-2",
                    "acceptable_pages": [7],
                    "anchor_terms": ["0.901"],
                },
            ],
        }
    )


def _response() -> dict:
    return {
        "answer": {
            "sufficient_evidence": True,
            "overview": "两篇论文采用不同模型。",
            "overview_evidence_ids": [1, 2],
            "claims": [{"statement": "结论", "evidence_ids": [2]}],
            "limitations": [],
        },
        "evidences": [
            {
                "evidence_id": 1,
                "document_id": "doc-1",
                "page_start": 3,
                "page_end": 3,
                "text": "使用注意力机制进行加权。",
            },
            {
                "evidence_id": 2,
                "document_id": "doc-2",
                "page_start": 6,
                "page_end": 8,
                "text": "最终识别率为 0.901。",
            },
        ],
    }


def test_complete_qa_response_passes() -> None:
    metrics = score_response(_case(), _response())

    assert metrics.sufficient_evidence_correct is True
    assert metrics.citation_validity == 1.0
    assert metrics.document_coverage == 1.0
    assert metrics.gold_page_recall == 1.0
    assert metrics.anchor_evidence_recall == 1.0
    assert case_passed(metrics) is True


def test_unknown_citation_fails() -> None:
    response = _response()
    response["answer"]["claims"][0]["evidence_ids"] = [99]

    metrics = score_response(_case(), response)

    assert metrics.invalid_citation_ids == [99]
    assert metrics.citation_validity == 2 / 3
    assert case_passed(metrics) is False


def test_missing_document_and_page_are_reported() -> None:
    response = _response()
    response["evidences"] = response["evidences"][:1]
    response["answer"]["overview_evidence_ids"] = [1]
    response["answer"]["claims"] = []

    metrics = score_response(_case(), response)

    assert metrics.document_coverage == 0.5
    assert metrics.missing_document_ids == ["doc-2"]
    assert metrics.gold_page_recall == 0.5
    assert metrics.missed_gold_evidence_indexes == [2]
    assert case_passed(metrics) is False
