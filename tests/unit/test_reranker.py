from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from researchpilot.rag.reranker import (
    RerankerService,
)


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict[str, Any] = field(
        default_factory=dict
    )


def make_point(
    chunk_id: str,
    document_id: str,
    text: str,
    score: float = 0.5,
    element_types: list[str] | None = None,
) -> FakePoint:
    return FakePoint(
        id=chunk_id,
        score=score,
        payload={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "text": text,
            "element_types": (
                element_types or ["text"]
            ),
        },
    )


class FakeCrossEncoder:
    """避免单元测试下载和加载真实模型。"""

    def __init__(
        self,
        scores: dict[str, float],
    ) -> None:
        self.scores = scores

    def predict(
        self,
        pairs,
        **kwargs,
    ):
        return np.empty(0) if not pairs else (
            np.asarray(
                [
                    self.scores[pair[1]]
                    for pair in pairs
                ],
                dtype=np.float32,
            )
        )


def test_reranker_orders_by_relevance():
    texts = [
        "低相关文本",
        "高相关文本",
        "中等相关文本",
    ]

    service = RerankerService(
        model=FakeCrossEncoder(
            {
                "低相关文本": 0.1,
                "高相关文本": 0.9,
                "中等相关文本": 0.5,
            }
        )
    )

    result = service.rerank(
        query="测试问题",
        texts=texts,
        top_k=2,
    )

    assert [
        item.text
        for item in result
    ] == [
        "高相关文本",
        "中等相关文本",
    ]

    assert result[0].original_index == 1
    assert result[0].score == pytest.approx(
        0.9
    )


def test_reranker_keeps_stable_order_for_ties():
    texts = [
        "第一条",
        "第二条",
    ]

    service = RerankerService(
        model=FakeCrossEncoder(
            {
                "第一条": 0.5,
                "第二条": 0.5,
            }
        )
    )

    result = service.rerank(
        query="测试问题",
        texts=texts,
        top_k=2,
    )

    assert [
        item.original_index
        for item in result
    ] == [0, 1]


def test_reranker_handles_empty_texts():
    service = RerankerService(
        model=FakeCrossEncoder({})
    )

    assert service.rerank(
        query="测试问题",
        texts=[],
        top_k=3,
    ) == []


def test_reranker_rejects_invalid_top_k():
    service = RerankerService(
        model=FakeCrossEncoder({})
    )

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        service.rerank(
            query="测试问题",
            texts=["证据"],
            top_k=0,
        )


def test_rerank_points_preserves_documents():
    points = [
        make_point(
            "doc-a-high",
            "doc-a",
            "文献A高相关证据",
        ),
        make_point(
            "doc-a-low",
            "doc-a",
            "文献A低相关证据",
        ),
        make_point(
            "doc-b-low",
            "doc-b",
            "文献B低相关证据",
        ),
    ]

    service = RerankerService(
        model=FakeCrossEncoder(
            {
                "文献A高相关证据": 0.9,
                "文献A低相关证据": 0.8,
                "文献B低相关证据": 0.1,
            }
        )
    )

    selected, score_map = (
        service.rerank_points(
            query="跨文献比较",
            points=points,
            top_k=2,
            document_ids=[
                "doc-a",
                "doc-b",
            ],
        )
    )

    selected_document_ids = {
        point.payload["document_id"]
        for point in selected
    }

    assert selected_document_ids == {
        "doc-a",
        "doc-b",
    }
    assert len(score_map) == 3


def test_rerank_points_preserves_table():
    points = [
        make_point(
            "normal",
            "doc-a",
            "普通正文",
        ),
        make_point(
            "table",
            "doc-a",
            "实验指标表格",
            element_types=["table"],
        ),
    ]

    service = RerankerService(
        model=FakeCrossEncoder(
            {
                "普通正文": 0.9,
                "实验指标表格": 0.1,
            }
        )
    )

    selected, _ = service.rerank_points(
        query="评价指标是什么",
        points=points,
        top_k=1,
        document_ids=["doc-a"],
        require_table=True,
    )

    assert (
        selected[0]
        .payload["chunk_id"]
        == "table"
    )


class FakeQueryAwareCrossEncoder:
    """根据问题和文本组合返回测试分数。"""

    def __init__(
        self,
        scores: dict[tuple[str, str], float],
    ) -> None:
        self.scores = scores

    def predict(
        self,
        pairs,
        **kwargs,
    ):
        return np.asarray(
            [
                self.scores[
                    (pair[0], pair[1])
                ]
                for pair in pairs
            ],
            dtype=np.float32,
        )


def test_reranker_uses_best_subquery_score():
    service = RerankerService(
        model=FakeQueryAwareCrossEncoder(
            {
                (
                    "比较两种方法",
                    "一般性摘要",
                ): 0.90,
                (
                    "比较两种方法",
                    "CNNSSD-SVM在10-shot下为92.52%",
                ): 0.10,
                (
                    "CNNSSD-SVM的10-shot结果是多少",
                    "一般性摘要",
                ): 0.20,
                (
                    "CNNSSD-SVM的10-shot结果是多少",
                    "CNNSSD-SVM在10-shot下为92.52%",
                ): 0.95,
            }
        )
    )

    result = service.rerank_queries(
        queries=[
            "比较两种方法",
            "CNNSSD-SVM的10-shot结果是多少",
        ],
        texts=[
            "一般性摘要",
            "CNNSSD-SVM在10-shot下为92.52%",
        ],
        top_k=1,
    )

    assert len(result) == 1
    assert (
        result[0].text
        == "CNNSSD-SVM在10-shot下为92.52%"
    )
    assert result[0].score == pytest.approx(
        0.95
    )