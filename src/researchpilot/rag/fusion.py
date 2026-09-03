from typing import Any


def point_key(point: Any) -> str:
    """返回 Qdrant 检索结果的稳定唯一标识。"""
    payload = point.payload or {}

    chunk_id = payload.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    return str(point.id)


def reciprocal_rank_fusion(
    rankings: list[list[Any]],
    rrf_k: int = 60,
) -> list[Any]:
    """
    使用 Reciprocal Rank Fusion 融合多组检索结果。

    同一个片段在越多检索结果中出现、排名越靠前，
    最终排序就越靠前。
    """
    if rrf_k < 1:
        raise ValueError("rrf_k 必须大于等于 1")

    scores: dict[str, float] = {}
    points: dict[str, Any] = {}

    for ranking in rankings:
        seen_in_ranking: set[str] = set()

        for rank, point in enumerate(ranking, start=1):
            key = point_key(point)

            # 防止同一个检索结果内部出现重复片段
            if key in seen_in_ranking:
                continue

            seen_in_ranking.add(key)
            points.setdefault(key, point)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    sorted_keys = sorted(
        scores,
        key=lambda key: (-scores[key], key),
    )

    return [points[key] for key in sorted_keys]

"""多样性融合函数"""
def reciprocal_rank_fusion_with_diversity(
    rankings: list[list[Any]],
    *,
    limit: int,
    extra_limit: int = 2,
    rrf_k: int = 60,
) -> list[Any]:
    """
    先保留 RRF 排名前 limit 的结果，
    再从各个查询的独有候选中补充最多 extra_limit 条。

    返回数量最多为 limit + extra_limit。
    """
    if limit < 1:
        raise ValueError("limit 必须大于等于 1")

    if extra_limit < 0:
        raise ValueError(
            "extra_limit 必须大于等于 0"
        )

    fused_points = reciprocal_rank_fusion(
        rankings,
        rrf_k=rrf_k,
    )

    selected_points = fused_points[:limit]

    if extra_limit == 0:
        return selected_points

    selected_keys = {
        point_key(point)
        for point in selected_points
    }

    # key -> (
    #     在单路查询中的排名,
    #     负向量分数,
    #     查询序号,
    #     point,
    # )
    diversity_candidates: dict[
        str,
        tuple[int, float, int, Any],
    ] = {}

    for query_index, ranking in enumerate(
        rankings
    ):
        for rank, point in enumerate(
            ranking,
            start=1,
        ):
            key = point_key(point)

            if key in selected_keys:
                continue

            score = float(
                getattr(point, "score", 0.0)
                or 0.0
            )

            candidate = (
                rank,
                -score,
                query_index,
                point,
            )

            existing = (
                diversity_candidates.get(key)
            )

            if (
                existing is None
                or candidate[:3] < existing[:3]
            ):
                diversity_candidates[key] = (
                    candidate
                )

            # 每一路查询只提出一条尚未进入
            # RRF 主结果的最佳候选
            break

    ordered_candidates = sorted(
        diversity_candidates.items(),
        key=lambda item: (
            item[1][0],
            item[1][1],
            item[1][2],
            item[0],
        ),
    )

    for key, candidate in ordered_candidates:
        if key in selected_keys:
            continue

        selected_points.append(candidate[3])
        selected_keys.add(key)

        if (
            len(selected_points)
            >= limit + extra_limit
        ):
            break

    return selected_points