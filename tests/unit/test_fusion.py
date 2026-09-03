from dataclasses import dataclass, field
from typing import Any

import pytest

from researchpilot.rag.fusion import (
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_with_diversity,
)


@dataclass
class FakePoint:
    id: str
    score: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


def make_point(chunk_id: str) -> FakePoint:
    return FakePoint(
        id=chunk_id,
        payload={"chunk_id": chunk_id},
    )


def test_rrf_promotes_results_found_by_multiple_queries():
    point_a = make_point("a")
    point_b = make_point("b")
    point_c = make_point("c")

    result = reciprocal_rank_fusion(
        rankings=[
            [point_a, point_b],
            [point_b, point_c],
        ],
        rrf_k=60,
    )

    assert result[0].id == "b"


def test_rrf_removes_duplicate_points():
    point_a = make_point("a")

    result = reciprocal_rank_fusion(
        rankings=[
            [point_a, point_a],
            [point_a],
        ],
        rrf_k=60,
    )

    assert len(result) == 1


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            rankings=[],
            rrf_k=0,
        )


def test_diversity_adds_query_specific_points():
    point_a = make_point("a")
    point_b = make_point("b")
    point_c = make_point("c")
    point_d = make_point("d")

    result = (
        reciprocal_rank_fusion_with_diversity(
            rankings=[
                [
                    point_a,
                    point_b,
                    point_c,
                ],
                [
                    point_a,
                    point_b,
                    point_d,
                ],
            ],
            limit=2,
            extra_limit=2,
            rrf_k=60,
        )
    )

    result_ids = [
        point.id
        for point in result
    ]

    assert result_ids[:2] == ["a", "b"]
    assert "c" in result_ids
    assert "d" in result_ids
    assert len(result_ids) == 4


def test_diversity_can_be_disabled():
    point_a = make_point("a")
    point_b = make_point("b")
    point_c = make_point("c")

    result = (
        reciprocal_rank_fusion_with_diversity(
            rankings=[
                [
                    point_a,
                    point_b,
                    point_c,
                ],
            ],
            limit=2,
            extra_limit=0,
            rrf_k=60,
        )
    )

    assert [
        point.id
        for point in result
    ] == ["a", "b"]


def test_diversity_rejects_invalid_limits():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion_with_diversity(
            rankings=[],
            limit=0,
        )

    with pytest.raises(ValueError):
        reciprocal_rank_fusion_with_diversity(
            rankings=[],
            limit=1,
            extra_limit=-1,
        )