import json
from pathlib import Path

import pytest

from researchpilot.evaluation.loader import load_eval_cases

VALID_CASE_DATA = {
    "case_id": "qa_001",
    "endpoint": "qa",
    "scope": "single_document",
    "question_type": "fact",
    "difficulty": "easy",
    "question": "论文使用了什么方法？",
    "document_ids": ["doc-1"],
    "answerability": "full",
    "expected_sufficient_evidence": True,
}


def _valid_case_json() -> str:
    return json.dumps(VALID_CASE_DATA, ensure_ascii=False)


def test_load_eval_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(_valid_case_json() + "\n", encoding="utf-8")

    cases = load_eval_cases(path)

    assert len(cases) == 1
    assert cases[0].case_id == "qa_001"
    assert cases[0].gold_evidence == []


def test_duplicate_case_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    case_json = _valid_case_json()
    path.write_text(case_json + "\n" + case_json + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="case_id 重复"):
        load_eval_cases(path)


def test_qa_case_requires_question(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    case_data = VALID_CASE_DATA.copy()
    case_data.pop("question")
    path.write_text(json.dumps(case_data, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="question"):
        load_eval_cases(path)
