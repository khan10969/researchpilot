from researchpilot.rag.service import (
    RAGService,
)


def test_table_question_is_detected() -> None:
    assert (
        RAGService._needs_table_evidence(
            "ABi-GRU在表2中的识别性能是多少？"
        )
        is True
    )


def test_metric_question_is_detected() -> None:
    assert (
        RAGService._needs_table_evidence(
            "哪一个10-shot识别率更高？"
        )
        is True
    )


def test_method_question_does_not_require_table() -> None:
    assert (
        RAGService._needs_table_evidence(
            "该方法包含哪些主要处理步骤？"
        )
        is False
    )


def test_metric_question_needs_table_evidence():
    assert RAGService._needs_table_evidence(
        "两篇论文的任务设置和评价指标有什么不同？"
    )


def test_evaluation_question_needs_table_evidence():
    assert RAGService._needs_table_evidence(
        "论文采用了哪些评价方法？"
    )