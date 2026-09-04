from pathlib import Path
from time import perf_counter
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from openai import APIError

##########################################
from researchpilot.analysis.experiment_service import (
    ExperimentAnalysisService,
)
from researchpilot.analysis.models import (
    DocumentTablesResponse,
    ExperimentAnalysisResult,
)
from researchpilot.analysis.table_service import (
    TableAnalysisService,
)
from researchpilot.api.dependencies import (
    get_document_service,
    get_experiment_analysis_service,
    get_ingestion_service,
    get_rag_service,
    get_table_analysis_service,
)
from researchpilot.api.schemas import (
    AskRequest,
    ExperimentAnalysisRequest,
    HealthResponse,
)
from researchpilot.config import settings

#####################
from researchpilot.documents.service import (
    DocumentService,
)
from researchpilot.ingestion.models import (
    DocumentDeleteResult,
    DocumentIndexResult,
    DocumentListResponse,
)
from researchpilot.ingestion.service import (
    DocumentIngestionService,
)
from researchpilot.llm_utils import (
    ModelOutputError,
)
from researchpilot.rag.models import RAGResult
from researchpilot.rag.service import RAGService
from researchpilot.storage.qdrant_store import QdrantStore

from researchpilot.observability import (
    bind_request_id,
    configure_logging,
    get_logger,
    reset_request_id,
    resolve_request_id,
)

configure_logging()
api_logger = get_logger(__name__)

app = FastAPI(
    title="ResearchPilot API",
    description=("面向科研文献的证据检索与可定位引用问答服务"),
    version="0.1.0",
)


######################################################################
@app.middleware("http")
async def observe_request(
    request: Request,
    call_next,
) -> Response:
    """为每个 HTTP 请求添加编号和耗时日志。"""
    request_id = resolve_request_id(request.headers.get("X-Request-ID"))
    token = bind_request_id(request_id)
    started = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round(
            (perf_counter() - started) * 1000,
            2,
        )

        api_logger.exception(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
        )
        raise
    else:
        duration_ms = round(
            (perf_counter() - started) * 1000,
            2,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["Server-Timing"] = f"app;dur={duration_ms}"

        if request.url.path != "/api/v1/health":
            api_logger.info(
                "request_completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=(response.status_code),
                duration_ms=duration_ms,
            )

        return response
    finally:
        reset_request_id(token)


#####################################################################


@app.get(
    "/",
    summary="服务首页",
)
def root() -> dict[str, str]:
    return {
        "name": "ResearchPilot API",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="检查服务状态",
)
def health() -> HealthResponse:
    store = QdrantStore()

    try:
        # 发起一次真实请求，确认 Qdrant 可访问
        store.client.get_collections()

        collection_exists = store.client.collection_exists(store.collection_name)

        point_count = store.count_points() if collection_exists else 0

        return HealthResponse(
            status="ok",
            qdrant_connected=True,
            collection_name=store.collection_name,
            collection_exists=collection_exists,
            point_count=point_count,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant 当前不可用：{exc}",
        ) from exc


@app.post(
    "/api/v1/qa/ask",
    response_model=RAGResult,
    summary="基于文献证据回答问题",
)
def ask_question(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> RAGResult:
    try:
        return service.answer(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except APIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"大模型服务调用失败：{exc}",
        ) from exc

    except ModelOutputError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post(
    "/api/v1/documents/upload",
    response_model=DocumentIndexResult,
    status_code=status.HTTP_201_CREATED,
    summary="上传并索引 PDF 文献",
)
def upload_document(
    file: UploadFile = File(...),
    service: DocumentIngestionService = Depends(get_ingestion_service),
) -> DocumentIndexResult:
    filename = Path(file.filename or "").name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="文件名不能为空",
        )

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="当前只支持 PDF 文件",
        )

    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    # 多读取一个字节，用于判断是否超过限制
    content = file.file.read(max_size_bytes + 1)

    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(f"文件过大，当前最大允许 {settings.max_upload_size_mb} MB"),
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="上传文件为空",
        )

    # 不能只相信扩展名，还要检查 PDF 文件头
    if not content.lstrip().startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="文件内容不是有效的 PDF",
        )

    try:
        return service.ingest_pdf_bytes(
            content=content,
            original_filename=filename,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get(
    "/api/v1/documents",
    response_model=DocumentListResponse,
    summary="获取文献列表",
)
def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    return service.list_documents()


@app.get(
    "/api/v1/documents/{document_id}",
    response_model=DocumentIndexResult,
    summary="获取单篇文献详情",
)
def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentIndexResult:
    try:
        return service.get_document(document_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.delete(
    "/api/v1/documents/{document_id}",
    response_model=DocumentDeleteResult,
    summary="删除文献及其向量",
)
def delete_document(
    document_id: str,
    confirm: bool = False,
    service: DocumentService = Depends(get_document_service),
) -> DocumentDeleteResult:
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail=("删除操作需要明确设置 confirm=true"),
        )

    try:
        return service.delete_document(document_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"删除文献失败：{exc}",
        ) from exc


@app.get(
    "/api/v1/documents/{document_id}/tables",
    response_model=DocumentTablesResponse,
    summary="获取文献中的结构化表格",
)
def get_document_tables(
    document_id: str,
    service: TableAnalysisService = Depends(get_table_analysis_service),
) -> DocumentTablesResponse:
    try:
        return service.get_document_tables(document_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.get(
    "/api/v1/documents/{document_id}/source",
    response_class=FileResponse,
    summary="查看文献原始 PDF",
)
def get_document_source(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> FileResponse:
    try:
        source_path = service.get_source_path(document_id)

        return FileResponse(
            path=source_path,
            media_type="application/pdf",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.post(
    "/api/v1/analysis/experiments",
    response_model=ExperimentAnalysisResult,
    summary="分析并比较文献实验结果",
)
def analyze_experiments(
    request: ExperimentAnalysisRequest,
    service: ExperimentAnalysisService = Depends(get_experiment_analysis_service),
) -> ExperimentAnalysisResult:
    try:
        return service.analyze(
            document_ids=request.document_ids,
            focus=request.focus,
            top_k_per_document=(request.top_k_per_document),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except APIError as exc:
        raise HTTPException(
            status_code=502,
            detail=(f"实验分析模型调用失败：{exc}"),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc
