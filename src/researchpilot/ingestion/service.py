import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from docling.document_converter import DocumentConverter

from researchpilot.config import settings
from researchpilot.ingestion.chunker import build_chunks
from researchpilot.ingestion.models import (
    DocumentIndexResult,
)
from researchpilot.retrieval.embeddings import (
    EmbeddingService,
)
from researchpilot.storage.qdrant_store import (
    QdrantStore,
)


class DocumentIngestionService:
    """负责 PDF 保存、解析、切块和向量索引。"""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        store: QdrantStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.store = store

        self.documents_dir = (
            settings.data_root / "documents"
        )

        self.incoming_dir = (
            settings.data_root / "incoming"
        )

        self.documents_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.incoming_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.converter = DocumentConverter()

        # 当前 MVP 每次只处理一篇上传文档，
        # 避免 Docling 和 GPU 同时执行多个任务。
        self._ingestion_lock = Lock()

    def ingest_pdf_bytes(
        self,
        content: bytes,
        original_filename: str,
    ) -> DocumentIndexResult:
        """导入一份 PDF 文件。"""

        if not content:
            raise ValueError("上传文件为空")

        safe_filename = Path(
            original_filename
        ).name

        temporary_path = (
            self.incoming_dir
            / f"{uuid4().hex}.pdf"
        )

        temporary_path.write_bytes(content)

        try:
            with self._ingestion_lock:
                return self._process_pdf(
                    pdf_path=temporary_path,
                    content=content,
                    original_filename=safe_filename,
                )

        finally:
            temporary_path.unlink(
                missing_ok=True
            )

    def _process_pdf(
        self,
        pdf_path: Path,
        content: bytes,
        original_filename: str,
    ) -> DocumentIndexResult:
        print(
            f"开始解析上传文件：{original_filename}"
        )

        conversion_result = (
            self.converter.convert(pdf_path)
        )

        document = conversion_result.document
        document_dict = document.export_to_dict()

        origin = document_dict.setdefault(
            "origin",
            {},
        )

        # Docling 当前解析的是临时文件名，
        # 因此将来源名称恢复为用户上传的文件名。
        origin["filename"] = original_filename

        binary_hash = origin.get("binary_hash")

        if binary_hash is not None:
            document_id = str(binary_hash)
        else:
            # 极少数情况下 Docling 没有返回 binary_hash，
            # 就使用标准 SHA-256。
            document_id = hashlib.sha256(
                content
            ).hexdigest()

            origin["binary_hash"] = document_id

        document_dir = (
            self.documents_dir / document_id
        )

        metadata_path = (
            document_dir / "metadata.json"
        )

        # 相同 PDF 再次上传时直接返回已有结果
        if metadata_path.exists():
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            existing_result = (
                DocumentIndexResult.model_validate(
                    metadata
                )
            )

            existing_result.duplicate = True

            print(
                f"文档已经索引，跳过重复处理："
                f"{document_id}"
            )

            return existing_result

        document_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_path = (
            document_dir / "source.pdf"
        )

        markdown_path = (
            document_dir / "document.md"
        )

        json_path = (
            document_dir / "document.json"
        )

        chunks_path = (
            document_dir / "chunks.json"
        )

        source_path.write_bytes(content)

        markdown = document.export_to_markdown()

        markdown_path.write_text(
            markdown,
            encoding="utf-8",
        )

        json_path.write_text(
            json.dumps(
                document_dict,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        chunks = build_chunks(document_dict)

        if not chunks:
            raise RuntimeError(
                "Docling 解析成功，但没有生成有效文本块"
            )

        # 确保 API 返回的 document_id、
        # chunk_id 和 Qdrant payload 完全一致。
        for index, chunk in enumerate(chunks):
            chunk.document_id = document_id
            chunk.chunk_index = index
            chunk.chunk_id = (
                f"{document_id}-{index:04d}"
            )
            chunk.source_file = original_filename

        chunks_path.write_text(
            json.dumps(
                [
                    chunk.model_dump(mode="json")
                    for chunk in chunks
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        embedding_texts = [
            f"章节：{chunk.section}\n{chunk.text}"
            for chunk in chunks
        ]

        print(
            f"开始生成向量，共 {len(chunks)} 个文本块"
        )

        embeddings = (
            self.embedding_service.encode_documents(
                embedding_texts
            )
        )

        self.store.ensure_collection(
            vector_size=(
                self.embedding_service.dimension
            )
        )

        print("开始写入 Qdrant")

        self.store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        page_count = len(
            document_dict.get("pages", {})
        )

        index_result = DocumentIndexResult(
            document_id=document_id,
            filename=original_filename,
            status="indexed",
            page_count=page_count,
            chunk_count=len(chunks),
            collection_name=(
                self.store.collection_name
            ),
            total_point_count=(
                self.store.count_points()
            ),
            indexed_at=datetime.now(
                timezone.utc
            ).isoformat(),
            duplicate=False,
        )

        metadata_path.write_text(
            json.dumps(
                index_result.model_dump(
                    mode="json"
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"文档索引完成：{document_id}"
        )

        return index_result