"""大模型结构化输出的公共容错工具。"""

from collections.abc import Callable
from typing import TypeVar

from pydantic import ValidationError

ResultType = TypeVar("ResultType")


class ModelOutputError(RuntimeError):
    """大模型多次返回无法解析或校验的结构化结果。"""


def retry_structured_output(
    operation: Callable[[], ResultType],
    *,
    attempts: int,
    label: str,
) -> ResultType:
    """
    当大模型返回的 JSON 无法解析或校验时，
    重新调用模型。

    attempts 表示总尝试次数，而不是额外重试次数。
    """

    last_error: Exception | None = None

    for _ in range(max(1, attempts)):
        try:
            return operation()

        except (ValidationError, RuntimeError) as exc:
            last_error = exc

    raise ModelOutputError(
        f"{label}连续 {max(1, attempts)} 次"
        "返回无效的结构化结果。"
    ) from last_error