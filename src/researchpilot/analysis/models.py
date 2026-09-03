from typing import Any, Literal

from pydantic import BaseModel, Field


class ExtractedTableCell(BaseModel):
    """Docling 表格中的一个单元格。"""

    text: str

    row_start: int
    row_end: int

    col_start: int
    col_end: int

    row_span: int = 1
    col_span: int = 1

    column_header: bool = False
    row_header: bool = False

    bbox: dict[str, Any] | None = None


class ExtractedTable(BaseModel):
    """从一篇论文中提取出的表格。"""

    table_id: str
    document_id: str
    source_file: str

    source_ref: str
    caption: str | None = None

    page_no: int | None = None
    bbox: dict[str, Any] | None = None

    num_rows: int
    num_cols: int

    # 适合前端直接渲染的二维数组
    rows: list[list[str]] = Field(
        default_factory=list
    )

    # 保留合并单元格和精确坐标的信息
    cells: list[ExtractedTableCell] = Field(
        default_factory=list
    )


class DocumentTablesResponse(BaseModel):
    """一篇文献的全部表格。"""

    document_id: str
    source_file: str

    table_count: int
    tables: list[ExtractedTable]


######################################################
class AnalysisEvidence(BaseModel):
    """实验分析使用的一条证据。"""

    evidence_id: int

    evidence_type: Literal[
        "text_chunk",
        "table",
    ]

    document_id: str
    source_file: str

    section: str
    page_start: int | None = None
    page_end: int | None = None

    retrieval_score: float | None = None

    text: str

    source_ref: str | None = None
    bbox: dict[str, Any] | None = None

    source_locations: list[
        dict[str, Any]
    ] = Field(default_factory=list)


class EvidenceBackedClaim(BaseModel):
    """一条必须附带证据的实验事实。"""

    statement: str = Field(
        min_length=1
    )

    evidence_ids: list[int] = Field(
        min_length=1
    )


class AnalysisLimitation(BaseModel):
    """
    实验分析中的限制。

    论文明确报告的局限性应带证据；如果限制来自“当前证据
    未提供某项信息”，则允许 evidence_ids 为空。
    """

    statement: str = Field(
        min_length=1
    )

    evidence_ids: list[int] = Field(
        default_factory=list
    )


class DocumentExperimentSummary(BaseModel):
    """单篇论文的实验信息摘要。"""

    document_id: str
    source_file: str

    research_tasks: list[
        EvidenceBackedClaim
    ]

    methods: list[
        EvidenceBackedClaim
    ]

    datasets: list[
        EvidenceBackedClaim
    ]

    baselines: list[
        EvidenceBackedClaim
    ]

    experiment_settings: list[
        EvidenceBackedClaim
    ]

    metrics: list[
        EvidenceBackedClaim
    ]

    results: list[
        EvidenceBackedClaim
    ]

    limitations: list[
        AnalysisLimitation
    ]


class CrossDocumentFinding(BaseModel):
    """一条跨文献比较结论。"""

    aspect: str
    finding: str

    evidence_ids: list[int] = Field(
        min_length=1
    )


class ExperimentAnalysis(BaseModel):
    """DeepSeek 返回的结构化实验分析。"""

    sufficient_evidence: bool

    document_summaries: list[
        DocumentExperimentSummary
    ]

    comparisons: list[
        CrossDocumentFinding
    ]

    overall_conclusion: str

    overall_evidence_ids: list[int]

    limitations: list[str]


class ExperimentAnalysisResult(BaseModel):
    """实验分析接口的完整响应。"""

    document_ids: list[str]
    focus: str | None

    analysis: ExperimentAnalysis
    evidences: list[AnalysisEvidence]
