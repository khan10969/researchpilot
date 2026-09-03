import re
from collections.abc import Iterator
from typing import Any

from researchpilot.ingestion.models import DocumentChunk, SourceLocation


def _resolve_ref(
        document:dict[str, Any],
        ref:str,
) -> dict[str, Any]:
    """
    将 '#/texts/10'、'#/tables/0' 这样的引用，
    转换成实际的字典对象。
    """

    node: Any = document

    for part in ref.removeprefix("#/").split("/"):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]

    return node


def _walk_document_items(
        document:dict[str, Any],
        node:dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """
    按照 Docling 记录的文档顺序遍历正文、表格等元素。
    """

    for child in node.get("children", []):
        ref = child.get("$ref")

        if not ref:
            continue

        item = _resolve_ref(document, ref)

        # group 本身通常没有正文，需要继续遍历其子元素
        if ref.startswith("#/groups/"):
            yield from _walk_document_items(document, item)
        else:
            yield item


def _normalize_text(text: str) -> str:
    """清楚多余空格，但尽量保留段落结构。"""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _table_to_text(table:dict[str, Any]) -> str:
    """
    将 Docling 表格转换为便于向量检索的纯文本形式。
    """

    grid = table.get("data",{}).get("grid",[])
    rows: list[str] = []

    for row in grid:
        cells = [
            _normalize_text(str(cell.get("text",""))) for cell in row
        ]

        if any(cells):
            rows.append(" | ".join(cells))

    if not rows:
        return ""

    return "[表格]\n" + "\n".join(rows)


def _get_locations(
    item: dict[str, Any],
) -> list[SourceLocation]:
    """提取页码和 PDF 坐标。"""

    locations: list[SourceLocation] = []

    for provenance in item.get("prov", []):
        page_no = provenance.get("page_no")

        if page_no is None:
            continue

        locations.append(
            SourceLocation(
                source_ref=item.get("self_ref", ""),
                page_no=int(page_no),
                bbox=provenance.get("bbox"),
            )
        )

    return locations


def _split_long_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """
    如果单个段落本身太长，就使用滑动窗口继续切分。
    """

    if len(text) <= max_chars:
        return [text]

    step = max_chars - overlap_chars

    if step <= 0:
        raise ValueError("overlap_chars 必须小于 max_chars")

    parts: list[str] = []

    for start in range(0, len(text), step):
        part = text[start : start + max_chars].strip()

        if part:
            parts.append(part)

        if start + max_chars >= len(text):
            break

    return parts


def build_chunks(
    document: dict[str, Any],
    max_chars: int = 1000,
    overlap_chars: int = 120,
) -> list[DocumentChunk]:
    """
    将一篇 Docling JSON 文档转换为多个 DocumentChunk。
    """

    origin = document.get("origin", {})

    source_file = origin.get(
        "filename",
        f"{document.get('name', 'unknown')}.pdf",
    )

    document_id = str(
        origin.get("binary_hash")
        or document.get("name")
        or source_file
    )

    # 每一项包含：
    # 章节名、元素类型、文本内容、来源位置
    elements: list[
        tuple[
            str,
            str,
            str,
            list[SourceLocation],
        ]
    ] = []

    current_section = "文档信息"

    for item in _walk_document_items(
        document,
        document.get("body", {}),
    ):
        if item.get("content_layer") != "body":
            continue

        label = item.get("label", "unknown")

        if label == "section_header":
            heading = _normalize_text(item.get("text", ""))

            if heading:
                current_section = heading

            continue

        if label == "table":
            text = _table_to_text(item)
        elif label in {
            "text",
            "list_item",
            "formula",
            "caption",
        }:
            text = _normalize_text(item.get("text", ""))
        else:
            continue

        if not text:
            continue

        locations = _get_locations(item)

        for text_part in _split_long_text(
            text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        ):
            elements.append(
                (
                    current_section,
                    label,
                    text_part,
                    locations,
                )
            )

    chunks: list[DocumentChunk] = []

    current_parts: list[str] = []
    current_types: list[str] = []
    current_locations: list[SourceLocation] = []
    active_section = "文档信息"

    def save_current_chunk() -> None:
        if not current_parts:
            return

        text = "\n\n".join(current_parts).strip()
        pages = sorted(
            {
                location.page_no
                for location in current_locations
            }
        )

        chunk_index = len(chunks)

        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}-{chunk_index:04d}",
                document_id=document_id,
                source_file=source_file,
                chunk_index=chunk_index,
                text=text,
                section=active_section,
                page_start=pages[0] if pages else None,
                page_end=pages[-1] if pages else None,
                element_types=sorted(set(current_types)),
                source_locations=current_locations.copy(),
            )
        )

        current_parts.clear()
        current_types.clear()
        current_locations.clear()

    for section, label, text, locations in elements:
        prospective_length = (
                sum(len(part) for part in current_parts)
                + len(text)
                + 2 * len(current_parts)
        )

        section_changed = (
                bool(current_parts)
                and section != active_section
        )

        size_exceeded = (
                bool(current_parts)
                and prospective_length > max_chars
        )

        if section_changed or size_exceeded:
            save_current_chunk()

        active_section = section
        current_parts.append(text)
        current_types.append(label)
        current_locations.extend(locations)

    save_current_chunk()

    return chunks



