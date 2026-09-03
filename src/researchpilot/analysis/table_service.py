import json
from typing import Any

from researchpilot.analysis.models import (
    DocumentTablesResponse,
    ExtractedTable,
    ExtractedTableCell,
)
from researchpilot.config import settings
from researchpilot.documents.service import (
    DocumentService,
)


class TableAnalysisService:
    """从 Docling JSON 中提取结构化表格。"""

    def __init__(
        self,
        document_service: DocumentService,
    ) -> None:
        self.document_service = document_service

    @staticmethod
    def _resolve_ref(
        document: dict[str, Any],
        ref: str,
    ) -> dict[str, Any]:
        """
        解析 '#/texts/10' 等 Docling 引用。
        """

        node: Any = document

        for part in ref.removeprefix(
            "#/"
        ).split("/"):
            if isinstance(node, list):
                node = node[int(part)]
            else:
                node = node[part]

        return node

    def _extract_caption(
        self,
        document: dict[str, Any],
        table: dict[str, Any],
    ) -> str | None:
        """提取与表格关联的标题。"""

        caption_parts: list[str] = []

        for caption_ref in table.get(
            "captions",
            [],
        ):
            ref = caption_ref.get("$ref")

            if not ref:
                continue

            try:
                caption_item = self._resolve_ref(
                    document,
                    ref,
                )

                text = str(
                    caption_item.get(
                        "text",
                        "",
                    )
                ).strip()

                if text:
                    caption_parts.append(text)

            except (
                KeyError,
                IndexError,
                ValueError,
            ):
                continue

        if not caption_parts:
            return None

        return " ".join(caption_parts)

    @staticmethod
    def _extract_rows(
        table: dict[str, Any],
    ) -> list[list[str]]:
        """生成适合直接展示的二维行列结构。"""

        grid = (
            table.get("data", {})
            .get("grid", [])
        )

        rows: list[list[str]] = []

        for raw_row in grid:
            row: list[str] = []

            for raw_cell in raw_row:
                cell = raw_cell or {}

                text = str(
                    cell.get("text", "")
                ).strip()

                row.append(text)

            rows.append(row)

        return rows

    @staticmethod
    def _extract_cells(
        table: dict[str, Any],
    ) -> list[ExtractedTableCell]:
        """提取包含坐标和跨度的单元格。"""

        raw_cells = (
            table.get("data", {})
            .get("table_cells", [])
        )

        cells: list[
            ExtractedTableCell
        ] = []

        for cell in raw_cells:
            cells.append(
                ExtractedTableCell(
                    text=str(
                        cell.get("text", "")
                    ).strip(),
                    row_start=int(
                        cell.get(
                            "start_row_offset_idx",
                            0,
                        )
                    ),
                    row_end=int(
                        cell.get(
                            "end_row_offset_idx",
                            0,
                        )
                    ),
                    col_start=int(
                        cell.get(
                            "start_col_offset_idx",
                            0,
                        )
                    ),
                    col_end=int(
                        cell.get(
                            "end_col_offset_idx",
                            0,
                        )
                    ),
                    row_span=int(
                        cell.get(
                            "row_span",
                            1,
                        )
                    ),
                    col_span=int(
                        cell.get(
                            "col_span",
                            1,
                        )
                    ),
                    column_header=bool(
                        cell.get(
                            "column_header",
                            False,
                        )
                    ),
                    row_header=bool(
                        cell.get(
                            "row_header",
                            False,
                        )
                    ),
                    bbox=cell.get("bbox"),
                )
            )

        return cells

    def get_document_tables(
        self,
        document_id: str,
    ) -> DocumentTablesResponse:
        """获取指定文献的全部表格。"""

        # 同时完成 document_id 安全检查和存在性检查
        metadata = (
            self.document_service.get_document(
                document_id
            )
        )

        document_json_path = (
            settings.data_root
            / "documents"
            / metadata.document_id
            / "document.json"
        )

        if not document_json_path.exists():
            raise FileNotFoundError(
                "文献的 document.json 不存在："
                f"{metadata.document_id}"
            )

        document = json.loads(
            document_json_path.read_text(
                encoding="utf-8"
            )
        )

        extracted_tables: list[
            ExtractedTable
        ] = []

        for index, table in enumerate(
            document.get("tables", [])
        ):
            provenance = table.get(
                "prov",
                [],
            )

            first_provenance = (
                provenance[0]
                if provenance
                else {}
            )

            rows = self._extract_rows(
                table
            )

            data = table.get("data", {})

            num_rows = int(
                data.get(
                    "num_rows",
                    len(rows),
                )
            )

            default_num_cols = max(
                (
                    len(row)
                    for row in rows
                ),
                default=0,
            )

            num_cols = int(
                data.get(
                    "num_cols",
                    default_num_cols,
                )
            )

            extracted_tables.append(
                ExtractedTable(
                    table_id=(
                        f"{metadata.document_id}"
                        f"-table-{index:03d}"
                    ),
                    document_id=(
                        metadata.document_id
                    ),
                    source_file=(
                        metadata.filename
                    ),
                    source_ref=str(
                        table.get(
                            "self_ref",
                            f"#/tables/{index}",
                        )
                    ),
                    caption=self._extract_caption(
                        document,
                        table,
                    ),
                    page_no=(
                        int(
                            first_provenance[
                                "page_no"
                            ]
                        )
                        if first_provenance.get(
                            "page_no"
                        )
                        is not None
                        else None
                    ),
                    bbox=first_provenance.get(
                        "bbox"
                    ),
                    num_rows=num_rows,
                    num_cols=num_cols,
                    rows=rows,
                    cells=self._extract_cells(
                        table
                    ),
                )
            )

        return DocumentTablesResponse(
            document_id=metadata.document_id,
            source_file=metadata.filename,
            table_count=len(
                extracted_tables
            ),
            tables=extracted_tables,
        )