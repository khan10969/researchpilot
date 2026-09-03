"""语义评测的数据结构。"""

from pydantic import BaseModel, Field


class CoverageVerdict(BaseModel):
    """Judge 对一个应覆盖项目的判断。"""

    criterion_id: str
    satisfied: bool
    reason: str = Field(min_length=1)


class ForbiddenVerdict(BaseModel):
    """Judge 对一个禁止性结论的判断。"""

    criterion_id: str
    violated: bool
    reason: str = Field(min_length=1)


class SemanticJudgment(BaseModel):
    """DeepSeek Judge 返回的结构化结果。"""

    gold_points: list[CoverageVerdict]
    expected_limitations: list[CoverageVerdict]
    forbidden_claims: list[ForbiddenVerdict]


class SemanticMetrics(BaseModel):
    """根据 Judge 判断确定性计算出的指标。"""

    gold_point_coverage: float | None = None
    limitation_coverage: float | None = None
    forbidden_claim_safety: float
    semantic_passed: bool


class SemanticCaseResult(BaseModel):
    """一条案例的完整语义测评结果。"""

    case_id: str
    judge_model: str
    judgment: SemanticJudgment
    metrics: SemanticMetrics