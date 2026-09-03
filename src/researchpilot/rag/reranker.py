"""对初步召回的文本进行二阶段重排序。"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from researchpilot.config import settings
from researchpilot.rag.fusion import point_key


@dataclass(frozen=True, slots=True)
class RankedText:
    """一条经过重排序的文本。"""

    original_index: int
    score: float
    text: str


class RerankerService:
    """使用 CrossEncoder 计算问题与证据的相关性。"""

    def __init__(
        self,
        model: Any | None = None,
    ) -> None:
        """
        model 参数主要用于单元测试。

        正常运行时不传入 model，
        服务会加载配置中的真实重排序模型。
        """
        if model is not None:
            self.model = model
            self.device = "injected"
            return

        requested_device = (
            settings.reranker_device
        )

        if (
            requested_device.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            print(
                "重排序器无法使用CUDA，"
                "自动改用CPU"
            )
            self.device = "cpu"
        else:
            self.device = requested_device

        print(
            "正在加载重排序模型："
            f"{settings.reranker_model}"
        )
        print(
            f"重排序模型运行设备：{self.device}"
        )

        self.model = CrossEncoder(
            settings.reranker_model,
            device=self.device,
            max_length=(
                settings.reranker_max_length
            ),
        )

    def score_texts(
        self,
        query: str,
        texts: Sequence[str],
    ) -> np.ndarray:
        """批量计算问题与文本之间的相关性。"""
        query = query.strip()

        if not query:
            raise ValueError("query 不能为空")

        if not texts:
            return np.empty(
                shape=(0,),
                dtype=np.float32,
            )

        pairs = [
            [query, text]
            for text in texts
        ]

        raw_scores = self.model.predict(
            pairs,
            batch_size=(
                settings.reranker_batch_size
            ),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scores = np.asarray(
            raw_scores,
            dtype=np.float32,
        ).reshape(-1)

        if len(scores) != len(texts):
            raise RuntimeError(
                "重排序模型返回的分数数量"
                "与输入文本数量不一致"
            )

        return scores

    # def rerank(
    #     self,
    #     query: str,
    #     texts: Sequence[str],
    #     top_k: int,
    # ) -> list[RankedText]:
    #     """按照相关性重新排列并截取文本。"""
    #     if top_k < 1:
    #         raise ValueError(
    #             "top_k 必须大于等于 1"
    #         )
    #
    #     if not texts:
    #         return []
    #
    #     scores = self.score_texts(
    #         query=query,
    #         texts=texts,
    #     )
    #
    #     ranked_indices = sorted(
    #         range(len(texts)),
    #         key=lambda index: (
    #             -float(scores[index]),
    #             index,
    #         ),
    #     )
    #
    #     return [
    #         RankedText(
    #             original_index=index,
    #             score=float(scores[index]),
    #             text=texts[index],
    #         )
    #         for index in ranked_indices[:top_k]
    #     ]

    def rerank_queries(
        self,
        queries: Sequence[str],
        texts: Sequence[str],
        top_k: int,
    ) -> list[RankedText]:
        """
        使用多个检索子问题对文本进行重排序。

        每条文本分别与所有子问题计算相关性，
        最终使用最高相关性分数。
        """
        if top_k < 1:
            raise ValueError(
                "top_k 必须大于等于 1"
            )

        if not texts:
            return []

        clean_queries = list(
            dict.fromkeys(
                query.strip()
                for query in queries
                if query.strip()
            )
        )

        if not clean_queries:
            raise ValueError(
                "queries 至少包含一个非空问题"
            )

        pairs = [
            [query, text]
            for query in clean_queries
            for text in texts
        ]

        raw_scores = self.model.predict(
            pairs,
            batch_size=(
                settings.reranker_batch_size
            ),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        score_matrix = np.asarray(
            raw_scores,
            dtype=np.float32,
        ).reshape(
            len(clean_queries),
            len(texts),
        )

        # 一条证据只要与某个检索子问题高度相关，
        # 就应该保留较高的重排序分数。
        scores = np.max(
            score_matrix,
            axis=0,
        )

        ranked_indices = sorted(
            range(len(texts)),
            key=lambda index: (
                -float(scores[index]),
                index,
            ),
        )

        return [
            RankedText(
                original_index=index,
                score=float(scores[index]),
                text=texts[index],
            )
            for index in ranked_indices[:top_k]
        ]

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        top_k: int,
    ) -> list[RankedText]:
        """使用单个问题进行重排序。"""
        return self.rerank_queries(
            queries=[query],
            texts=texts,
            top_k=top_k,
        )


    @staticmethod
    def _document_id(point: Any) -> str:
        payload = point.payload or {}
        return str(
            payload.get("document_id", "")
        )

    @staticmethod
    def _is_table(point: Any) -> bool:
        payload = point.payload or {}

        element_types = payload.get(
            "element_types",
            [],
        )

        if isinstance(element_types, str):
            element_types = [element_types]

        return "table" in element_types

    @classmethod
    def _take_with_table_coverage(
        cls,
        ranked_points: list[Any],
        limit: int,
        require_table: bool,
    ) -> list[Any]:
        """
        选择排名靠前的结果。

        如果问题需要表格，且候选中存在表格，
        则至少保留一条排名最高的表格。
        """
        selected = ranked_points[:limit]

        if (
            not require_table
            or not ranked_points
            or any(
                cls._is_table(point)
                for point in selected
            )
        ):
            return selected

        best_table = next(
            (
                point
                for point in ranked_points
                if cls._is_table(point)
            ),
            None,
        )

        if best_table is None:
            return selected

        if selected:
            selected[-1] = best_table
        else:
            selected.append(best_table)

        return selected

    def rerank_points(
            self,
            query: str,
            points: Sequence[Any],
            top_k: int,
            document_ids: Sequence[str] | None = None,
            require_table: bool = False,
            queries: Sequence[str] | None = None,
    ) -> tuple[list[Any], dict[str, float]]:
        """
        对Qdrant候选点进行重排序。

        返回：
        1. 截取后的候选点；
        2. chunk_id到重排序分数的映射。
        """
        if top_k < 1:
            raise ValueError(
                "top_k 必须大于等于 1"
            )

        if not points:
            return [], {}

        # 防止同一个chunk重复进入重排序
        unique_points: list[Any] = []
        seen_keys: set[str] = set()

        for point in points:
            key = point_key(point)

            if key in seen_keys:
                continue

            seen_keys.add(key)
            unique_points.append(point)

        texts = [
            str(
                (point.payload or {}).get(
                    "text",
                    "",
                )
            )
            for point in unique_points
        ]

        # ranked_texts = self.rerank(
        #     query=query,
        #     texts=texts,
        #     top_k=len(texts),
        # )
        ranked_texts = self.rerank_queries(
            queries=queries or [query],
            texts=texts,
            top_k=len(texts),
        )

        ranked_points = [
            unique_points[item.original_index]
            for item in ranked_texts
        ]

        score_by_key = {
            point_key(
                unique_points[
                    item.original_index
                ]
            ): item.score
            for item in ranked_texts
        }

        clean_document_ids = list(
            dict.fromkeys(document_ids or [])
        )

        # 未限定文献或只有一篇文献
        if len(clean_document_ids) <= 1:
            selected = (
                self._take_with_table_coverage(
                    ranked_points=ranked_points,
                    limit=top_k,
                    require_table=require_table,
                )
            )

            return selected, score_by_key

        # 多篇文献时继续保证每篇都有名额
        effective_limit = max(
            top_k,
            len(clean_document_ids),
        )

        base_quota, remainder = divmod(
            effective_limit,
            len(clean_document_ids),
        )

        selected_points: list[Any] = []

        for index, document_id in enumerate(
            clean_document_ids
        ):
            document_quota = (
                base_quota
                + (1 if index < remainder else 0)
            )

            document_points = [
                point
                for point in ranked_points
                if self._document_id(point)
                == document_id
            ]

            document_selected = (
                self._take_with_table_coverage(
                    ranked_points=document_points,
                    limit=document_quota,
                    require_table=require_table,
                )
            )

            selected_points.extend(
                document_selected
            )

        # 某篇文献候选不足时，从全局排名中补齐
        selected_keys = {
            point_key(point)
            for point in selected_points
        }

        for point in ranked_points:
            if len(selected_points) >= effective_limit:
                break

            key = point_key(point)

            if key in selected_keys:
                continue

            selected_points.append(point)
            selected_keys.add(key)

        # 最终展示顺序按照重排序分数排列
        selected_points.sort(
            key=lambda point: (
                -score_by_key[point_key(point)],
                point_key(point),
            )
        )

        return (
            selected_points[:effective_limit],
            score_by_key,
        )