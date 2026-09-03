import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from researchpilot.config import settings
from researchpilot.ingestion.models import (
    DocumentDeleteResult,
    DocumentIndexResult,
    DocumentListResponse,
)
from researchpilot.storage.qdrant_store import (
    QdrantStore,
)

DOCUMENT_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{1,128}$"
)


class DocumentService:
    """管理已经导入的本地文献。"""

    def __init__(
        self,
        store: QdrantStore,
    ) -> None:
        self.store = store

        self.documents_dir = (
            settings.data_root / "documents"
        )

        self.trash_dir = (
            settings.data_root / "trash"
        )

        self.documents_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.trash_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _validate_document_id(
        document_id: str,
    ) -> str:
        document_id = document_id.strip()

        if not DOCUMENT_ID_PATTERN.fullmatch(
            document_id
        ):
            raise ValueError(
                "document_id 格式不合法"
            )

        return document_id

    def _get_document_dir(
        self,
        document_id: str,
    ) -> Path:
        document_id = (
            self._validate_document_id(
                document_id
            )
        )

        path = (
            self.documents_dir / document_id
        ).resolve()

        documents_root = (
            self.documents_dir.resolve()
        )

        # 防止 ../../ 等路径穿越
        if path.parent != documents_root:
            raise ValueError(
                "document_id 路径不合法"
            )

        return path

    def list_documents(
        self,
    ) -> DocumentListResponse:
        """列出所有已索引文献。"""

        documents: list[
            DocumentIndexResult
        ] = []

        for metadata_path in (
            self.documents_dir.glob(
                "*/metadata.json"
            )
        ):
            try:
                data = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )

                document = (
                    DocumentIndexResult.model_validate(
                        data
                    )
                )

                documents.append(document)

            except Exception as exc:
                print(
                    f"跳过损坏的 metadata："
                    f"{metadata_path}，原因：{exc}"
                )

        documents.sort(
            key=lambda item: item.indexed_at,
            reverse=True,
        )

        return DocumentListResponse(
            total=len(documents),
            items=documents,
        )

    def get_document(
        self,
        document_id: str,
    ) -> DocumentIndexResult:
        """获取单篇文献详情。"""

        document_dir = (
            self._get_document_dir(document_id)
        )

        metadata_path = (
            document_dir / "metadata.json"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"文献不存在：{document_id}"
            )

        data = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        return (
            DocumentIndexResult.model_validate(
                data
            )
        )

    def delete_document(
        self,
        document_id: str,
    ) -> DocumentDeleteResult:
        """
        删除 Qdrant 向量，并将本地目录移动到 trash。

        本地文件不会立即永久删除。
        """

        document = self.get_document(
            document_id
        )

        document_dir = (
            self._get_document_dir(document_id)
        )

        deleted_point_count = (
            self.store.count_document_points(
                document_id
            )
        )

        # Qdrant 中的删除不可直接撤销，
        # 但本地 source.pdf 会保留在 trash 中。
        self.store.delete_document_points(
            document_id
        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )

        trash_entry = (
            f"{document_id}-{timestamp}"
        )

        trash_path = (
            self.trash_dir / trash_entry
        )

        shutil.move(
            str(document_dir),
            str(trash_path),
        )

        if self.store.client.collection_exists(
            self.store.collection_name
        ):
            remaining_point_count = (
                self.store.count_points()
            )
        else:
            remaining_point_count = 0

        return DocumentDeleteResult(
            document_id=document_id,
            filename=document.filename,
            status="deleted",
            deleted_point_count=(
                deleted_point_count
            ),
            remaining_point_count=(
                remaining_point_count
            ),
            trash_entry=trash_entry,
        )

    def get_source_path(
        self,
        document_id: str,
    ) -> Path:
        """获取原始 PDF 的安全路径。"""

        document = self.get_document(
            document_id
        )

        source_path = (
            self.documents_dir
            / document.document_id
            / "source.pdf"
        ).resolve()

        if not source_path.exists():
            raise FileNotFoundError(
                f"原始 PDF 不存在："
                f"{document_id}"
            )

        return source_path