from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceLocation(BaseModel):
    """一段原文在PDF中的位置"""

    source_ref: str
    page_no: int
    bbox: dict[str, Any] | None = None


class DocumentChunk(BaseModel):
    """最终用于向量检索的文本块"""

    chunk_id: str
    document_id: str
    source_file: str
    chunk_index:int

    text:str
    section:str
    page_start: int | None
    page_end: int | None

    element_types: list[str] = Field(default_factory=list)
    source_locations: list[SourceLocation] = Field(default_factory=list)


class DocumentIndexResult(BaseModel):
    """一次文档导入与索引的结果。"""

    document_id: str
    filename: str

    status: Literal["indexed"]

    page_count: int
    chunk_count: int

    collection_name: str
    total_point_count: int

    indexed_at: str

    # 相同文件再次上传时为 true
    duplicate: bool = False


class DocumentListResponse(BaseModel):
    """文献列表响应。"""

    total: int
    items: list[DocumentIndexResult]


class DocumentDeleteResult(BaseModel):
    """删除文献的结果。"""

    document_id: str
    filename: str

    status: Literal["deleted"]

    deleted_point_count: int
    remaining_point_count: int

    # 本地文件没有永久删除，而是移动到这里
    trash_entry: str