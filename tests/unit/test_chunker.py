from typing import Any

from researchpilot.ingestion.chunker import (
    _table_to_text,
    build_chunks,
)


def make_document(
    body_text: str,
) -> dict[str, Any]:
    return {
        "name": "test-paper",
        "origin": {
            "filename": "test.pdf",
            "binary_hash": 123456,
        },
        "body": {
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
            ]
        },
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "page_header",
                "content_layer": "furniture",
                "text": "不应进入正文的页眉",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": None,
                    }
                ],
            },
            {
                "self_ref": "#/texts/1",
                "label": "section_header",
                "content_layer": "body",
                "text": "实验结果",
                "prov": [
                    {
                        "page_no": 2,
                        "bbox": None,
                    }
                ],
            },
            {
                "self_ref": "#/texts/2",
                "label": "text",
                "content_layer": "body",
                "text": body_text,
                "prov": [
                    {
                        "page_no": 2,
                        "bbox": {
                            "l": 10,
                            "t": 20,
                            "r": 100,
                            "b": 5,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    }
                ],
            },
        ],
        "groups": [],
        "tables": [],
    }


def test_chunk_preserves_section_and_page() -> None:
    document = make_document(
        "该方法在测试集上取得了较好的结果。"
    )

    chunks = build_chunks(
        document,
        max_chars=100,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.document_id == "123456"
    assert chunk.source_file == "test.pdf"
    assert chunk.section == "实验结果"
    assert chunk.page_start == 2
    assert chunk.page_end == 2
    assert chunk.text == (
        "该方法在测试集上取得了较好的结果。"
    )

    assert "不应进入正文的页眉" not in chunk.text
    assert (
        chunk.source_locations[0].source_ref
        == "#/texts/2"
    )


def test_long_text_has_overlap() -> None:
    document = make_document(
        "甲" * 250
    )

    chunks = build_chunks(
        document,
        max_chars=100,
        overlap_chars=20,
    )

    assert len(chunks) == 3
    assert all(
        len(chunk.text) <= 100
        for chunk in chunks
    )

    assert (
        chunks[0].text[-20:]
        == chunks[1].text[:20]
    )


def test_table_does_not_stop_after_empty_row() -> None:
    table = {
        "data": {
            "grid": [
                [
                    {"text": ""},
                    {"text": ""},
                ],
                [
                    {"text": "方法"},
                    {"text": "准确率"},
                ],
            ]
        }
    }

    result = _table_to_text(table)

    assert "[表格]" in result
    assert "方法 | 准确率" in result