"""将复杂科研问题拆解为多个检索子问题。"""

from openai import OpenAI
from pydantic import BaseModel, Field

from researchpilot.config import settings
from researchpilot.llm_utils import (
    retry_structured_output,
)

PLANNER_INSTRUCTIONS = """
你是科研文献检索规划器。

你的任务是把复杂问题拆成2至5个可以独立检索的子问题，
而不是回答问题。

必须遵守：

1. 不得回答用户问题。
2. 不得使用外部知识补充问题中没有出现的事实。
3. 必须保留方法名、模型名、指标名、数值条件和实验条件。
4. 跨文献比较问题应分别覆盖各方法以及需要比较的维度。
5. 如果问题包含多个明确要求，每个要求至少对应一个子问题。
6. 子问题应当能够独立用于向量检索，避免“它”“该方法”等指代。
7. 不要生成宽泛的“介绍该方法”类问题。
8. 子问题之间尽量减少重复。
9. 不得猜测原问题中没有出现的具体指标名、数据集名、算法名、数值或实验设置。
10. 可以使用原问题术语的通用同义表达，但不得把可能存在的指标当成确定事实。例如原问题只说“评价指标”时，应询问“采用了哪些评价指标”，不能自行加入 mAP、AUC 等具体指标。
11. 子问题只能细化原始问题中已经存在的要求，不能扩大用户的任务范围。
""".strip()


class QueryPlan(BaseModel):
    """大模型返回的问题拆解结果。"""

    subqueries: list[str] = Field(
        min_length=2,
        max_length=5,
    )


class QueryPlanner:
    """生成用于多路检索的查询计划。"""

    def __init__(
        self,
        client: OpenAI | None = None,
    ) -> None:
        self._owns_client = client is None

        if client is not None:
            self.client = client
            return

        if settings.llm_api_key is None:
            raise RuntimeError(
                "没有配置 LLM_API_KEY"
            )

        self.client = OpenAI(
            api_key=(
                settings.llm_api_key
                .get_secret_value()
            ),
            base_url=settings.llm_base_url,
            timeout=120.0,
            max_retries=2,
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    @staticmethod
    def should_decompose(
        document_count: int,
    ) -> bool:
        """
        初版只拆解跨文献问题。

        单篇问题继续使用原检索流程，
        避免增加无意义的模型调用。
        """

        return document_count > 1

    @staticmethod
    def _deduplicate_queries(
        original_query: str,
        subqueries: list[str],
    ) -> list[str]:
        queries: list[str] = []
        seen: set[str] = set()

        for value in [
            original_query,
            *subqueries,
        ]:
            cleaned = " ".join(
                value.strip().split()
            )

            if not cleaned:
                continue

            normalized = cleaned.casefold()

            if normalized in seen:
                continue

            seen.add(normalized)
            queries.append(cleaned)

        return queries


    @classmethod
    def fallback_plan(
            cls,
            query: str,
    ) -> list[str]:
        """
        大模型规划失败时使用的确定性备用计划。

        这些查询只用于扩大召回，不会被当成用户新增要求。
        """
        return cls._deduplicate_queries(
            original_query=query,
            subqueries=[
                f"{query} 方法 技术路线 模型结构",
                f"{query} 数据集 样本 实验设置",
                f"{query} 评价指标 实验结果 局限性",
            ],
        )


    def plan(
        self,
        query: str,
        document_count: int,
    ) -> list[str]:
        """
        返回原问题和拆解出的子问题。

        第一项始终保留原问题，避免拆解遗漏后
        完全丢失原始检索意图。
        """

        query = query.strip()

        if not query:
            raise ValueError("问题不能为空")

        if not self.should_decompose(
            document_count
        ):
            return [query]

        user_input = f"""
用户原始问题：

{query}

指定文献数量：

{document_count}

请生成2至5个检索子问题。
""".strip()

        def generate_plan() -> QueryPlan:
            response = (
                self.client.responses.parse(
                    model=settings.llm_model,
                    instructions=(
                        PLANNER_INSTRUCTIONS
                    ),
                    input=user_input,
                    text_format=QueryPlan,
                    max_output_tokens=800,
                    temperature=settings.llm_temperature,
                    reasoning={
                        "effort": "none",
                    },
                )
            )

            parsed = response.output_parsed

            if parsed is None:
                raise RuntimeError(
                    "问题规划器没有返回"
                    "可解析结果"
                )

            return parsed

        plan = retry_structured_output(
            generate_plan,
            attempts=(
                settings
                .llm_structured_max_attempts
            ),
            label="问题规划",
        )

        return self._deduplicate_queries(
            original_query=query,
            subqueries=plan.subqueries,
        )