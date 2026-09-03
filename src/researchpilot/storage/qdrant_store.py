import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from researchpilot.config import settings
from researchpilot.ingestion.models import DocumentChunk


class QdrantStore:
    """管理Qdrant collection，写入和检索。"""

    def __init__(self) -> None:
        self.collection_name = settings.qdrant_collection

        self.client = QdrantClient(
            url=settings.qdrant_url,
        )

    def ensure_collection(
            self,
            vector_size:int,
    ) -> None:
        """
       collection 不存在时创建；
       已存在时检查向量维度。
       """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name = self.collection_name,
                vectors_config = VectorParams(
                    size = vector_size,
                    distance=Distance.COSINE,
                )
            )

            print(
                f"已创建Qdrant_collection:"
                f"{self.collection_name}"
            )

            return

        collection = self.client.get_collection(self.collection_name)

        vectors_config = collection.config.params.vectors
        existing_size = getattr(
            vectors_config,
            "size",
            None,
        )

        if (
            existing_size is not None
            and existing_size != vector_size
        ):
            raise ValueError(
                "Qdrant collection 向量维度不匹配："
                f"现有维度={existing_size}，"
                f"当前模型维度={vector_size}"
            )

        print(
            f"使用已有 Qdrant collection："
            f"{self.collection_name}"
        )


    def upsert_chunks(
            self,
            chunks:Sequence[DocumentChunk],
            embeddings:np.ndarray,
    )->None:
        """将文本块及其向量写入 Qdrant。"""

        if len(chunks) != len(embeddings):
            raise ValueError("文本块数量与向量数量不一致")

        points: list[PointStruct] = []

        for chunk,embedding in zip(chunks, embeddings,strict=True):
            # Qdrant 的字符串 ID 必须是 UUID。
            # 使用 UUID5 可以保证重复执行时 ID 不变。
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    chunk.chunk_id,
                )
            )
            payload = chunk.model_dump(mode="json")

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def count_points(self)->int:
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return int(result.count)

    # def search(
    #         self,
    #         query_vector:np.ndarray,
    #         limit:int=5,
    # )->list[Any]:
    #     """搜索最相似的文本块。"""
    #
    #     response = self.client.query_points(
    #         collection_name = self.collection_name,
    #         query = query_vector.tolist(),
    #         limit = limit,
    #         with_payload = True,
    #         with_vectors = False,
    #     )
    #
    #     return list(response.points)

    # def search(
    #     self,
    #     query_vector: np.ndarray,
    #     limit: int = 5,
    #     document_ids: Sequence[str] | None = None,
    # ) -> list[Any]:
    #     """搜索最相似的文本块。"""
    #
    #     query_filter = None
    #
    #     if document_ids:
    #         query_filter = (
    #             self._build_document_filter(
    #                 document_ids
    #             )
    #         )
    #
    #     response = self.client.query_points(
    #         collection_name=self.collection_name,
    #         query=query_vector.tolist(),
    #         query_filter=query_filter,
    #         limit=limit,
    #         with_payload=True,
    #         with_vectors=False,
    #     )
    #
    #     return list(response.points)

    def search(
            self,
            query_vector: np.ndarray,
            limit: int = 5,
            document_ids: Sequence[str] | None = None,
            element_type: str | None = None,
    ) -> list[Any]:
        """
        搜索最相似的文本块。

        element_type 可用于只检索表格等指定类型。
        """

        must_conditions: list[Any] = []

        if document_ids:
            document_filter = (
                self._build_document_filter(
                    document_ids
                )
            )

            document_conditions = (
                document_filter.must
            )

            if isinstance(
                    document_conditions,
                    list,
            ):
                must_conditions.extend(
                    document_conditions
                )
            elif document_conditions is not None:
                must_conditions.append(
                    document_conditions
                )

        if element_type:
            must_conditions.append(
                FieldCondition(
                    key="element_types",
                    match=MatchValue(
                        value=element_type
                    ),
                )
            )

        query_filter = (
            Filter(must=must_conditions)
            if must_conditions
            else None
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return list(response.points)





    @staticmethod
    def _build_document_filter(
            document_ids: Sequence[str],
    ) -> Filter:
        """构造 document_id 过滤条件。"""

        cleaned_ids = list(
            dict.fromkeys(
                str(document_id)
                for document_id in document_ids
            )
        )

        if not cleaned_ids:
            raise ValueError(
                "document_ids 不能为空"
            )

        if len(cleaned_ids) == 1:
            match = MatchValue(
                value=cleaned_ids[0]
            )
        else:
            match = MatchAny(
                any=cleaned_ids
            )

        return Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=match,
                )
            ]
        )



    def count_document_points(
        self,
        document_id: str,
    ) -> int:
        """统计某篇文献对应的向量数量。"""

        if not self.client.collection_exists(
            self.collection_name
        ):
            return 0

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=(
                self._build_document_filter(
                    [document_id]
                )
            ),
            exact=True,
        )

        return int(result.count)

    def delete_document_points(
        self,
        document_id: str,
    ) -> None:
        """删除某篇文献的全部向量。"""

        if not self.client.collection_exists(
            self.collection_name
        ):
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=(
                self._build_document_filter(
                    [document_id]
                )
            ),
            wait=True,
        )




