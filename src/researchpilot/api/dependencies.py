# from functools import lru_cache
#
# from researchpilot.rag.service import RAGService
#
#
# @lru_cache(maxsize=1)
# def get_rag_service() -> RAGService:
#     """
#     整个后端进程复用同一个 RAGService。
#
#     这样 BGE-M3 不会在每次请求时重新加载。
#     """
#     return RAGService()


from functools import lru_cache

from researchpilot.analysis.experiment_service import (
    ExperimentAnalysisService,
)
from researchpilot.analysis.table_service import (
    TableAnalysisService,
)
from researchpilot.documents.service import (
    DocumentService,
)
from researchpilot.ingestion.service import (
    DocumentIngestionService,
)
from researchpilot.rag.service import RAGService
from researchpilot.retrieval.embeddings import (
    EmbeddingService,
)
from researchpilot.storage.qdrant_store import (
    QdrantStore,
)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """整个后端只加载一份 BGE-M3。"""

    return EmbeddingService()


@lru_cache(maxsize=1)
def get_qdrant_store() -> QdrantStore:
    """复用 Qdrant 客户端。"""

    return QdrantStore()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """复用问答服务。"""

    return RAGService(
        embedding_service=(
            get_embedding_service()
        ),
        store=get_qdrant_store(),
    )


@lru_cache(maxsize=1)
def get_ingestion_service(
) -> DocumentIngestionService:
    """复用文档导入服务。"""

    return DocumentIngestionService(
        embedding_service=(
            get_embedding_service()
        ),
        store=get_qdrant_store(),
    )

@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    """复用文档管理服务。"""

    return DocumentService(
        store=get_qdrant_store()
    )


@lru_cache(maxsize=1)
def get_table_analysis_service(
) -> TableAnalysisService:
    """复用表格分析服务。"""

    return TableAnalysisService(
        document_service=(
            get_document_service()
        )
    )

####注册实验分析服务
@lru_cache(maxsize=1)
def get_experiment_analysis_service(
) -> ExperimentAnalysisService:
    """复用实验结果分析服务。"""

    return ExperimentAnalysisService(
        embedding_service=(
            get_embedding_service()
        ),
        store=get_qdrant_store(),
        document_service=(
            get_document_service()
        ),
        table_service=(
            get_table_analysis_service()
        ),
    )
