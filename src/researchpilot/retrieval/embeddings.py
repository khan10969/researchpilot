from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from researchpilot.config import settings


class EmbeddingService:
    """负责把文档和问题转换为向量"""

    def __init__(self) -> None:
        requested_device = settings.embedding_device

        if (
            requested_device.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            print("CUDA 当前不可用，自动改用CPU")
            self.device = "cpu"
        else:
            self.device = requested_device


        print(f"正在加载嵌入模型：{settings.embedding_model}")
        print(f"运行设备：{self.device}")

        self.model = SentenceTransformer(
            settings.embedding_model,
            device=self.device,
        )

        dimension = self.model.get_sentence_embedding_dimension()

        if dimension is None:
            raise RuntimeError("无法确定嵌入向量维度")

        self.dimension = int(dimension)

    def encode_documents(
            self,
            texts: Sequence[str],
    ) -> np.ndarray:
        """批量生成文档向量"""

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            list(texts),
            batch_size = settings.embedding_batch_size,
            show_progress_bar = True,
            convert_to_numpy=True,
            normalize_embeddings = True,
        )

        return embeddings.astype(np.float32)

    # def encode_query(self, query: str) -> np.ndarray:
    #     """生成单个查询问题的向量。"""
    #
    #     embedding = self.model.encode(
    #         query,
    #         show_progress_bar=False,
    #         convert_to_numpy=True,
    #         normalize_embeddings=True,
    #     )
    #
    #     return embedding.astype(np.float32)

    def encode_query(self, query: str):
        return self.encode_queries([query])[0]

    def encode_queries(self, queries: list[str]):
        if not queries:
            raise ValueError("queries 不能为空")

        return self.model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )