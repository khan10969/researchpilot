import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """文献问答请求"""

    query:str = Field(
        min_length=2,
        max_length=2000,
        description="需要依据文献回答的问题",
    )

    top_k: int | None = Field(
        default = None,
        ge = 1,
        le = 20,
        description="召回的证据数量",
    )

    document_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
        description=(
            "限定检索的文献 ID；"
            "不填写时检索全部文献"
        ),
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("问题不能为空")

        return value

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(
        cls,
        values: list[str] | None,
    ) -> list[str] | None:
        if values is None:
            return None

        cleaned: list[str] = []

        for value in values:
            document_id = value.strip()

            if not re.fullmatch(
                r"[A-Za-z0-9_-]{1,128}",
                document_id,
            ):
                raise ValueError(
                    f"document_id 格式不合法："
                    f"{value}"
                )

            if document_id not in cleaned:
                cleaned.append(document_id)

        return cleaned


class HealthResponse(BaseModel):
    """服务健康状态"""

    status: Literal["ok", "degraded"]

    qdrant_connected: bool
    collection_name: str
    collection_exists: bool
    point_count: int

    llm_provider: str
    llm_model: str


######################################
class ExperimentAnalysisRequest(BaseModel):
    """实验结果分析请求。"""

    document_ids: list[str] = Field(
        min_length=1,
        max_length=5,
        description="需要分析或比较的文献 ID",
    )

    focus: str | None = Field(
        default=None,
        max_length=500,
        description="用户希望重点比较的内容",
    )

    top_k_per_document: int | None = Field(
        default=None,
        ge=4,
        le=15,
        description="每篇文献召回的正文证据数",
    )

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(
        cls,
        values: list[str],
    ) -> list[str]:
        cleaned: list[str] = []

        for value in values:
            document_id = value.strip()

            if not re.fullmatch(
                r"[A-Za-z0-9_-]{1,128}",
                document_id,
            ):
                raise ValueError(
                    "document_id 格式不合法："
                    f"{value}"
                )

            if document_id not in cleaned:
                cleaned.append(document_id)

        return cleaned

    @field_validator("focus")
    @classmethod
    def validate_focus(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None