from researchpilot.rag.query_planner import (
    QueryPlanner,
)


def test_multi_document_question_is_decomposed() -> None:
    assert (
        QueryPlanner.should_decompose(2)
        is True
    )


def test_single_document_question_is_not_decomposed() -> None:
    assert (
        QueryPlanner.should_decompose(1)
        is False
    )


def test_queries_are_deduplicated() -> None:
    queries = (
        QueryPlanner._deduplicate_queries(
            original_query="比较两种方法",
            subqueries=[
                "比较两种方法",
                "  方法A使用什么数据？  ",
                "方法B使用什么数据？",
            ],
        )
    )

    assert queries == [
        "比较两种方法",
        "方法A使用什么数据？",
        "方法B使用什么数据？",
    ]