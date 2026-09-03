import pytest

from researchpilot.llm_utils import (
    ModelOutputError,
    retry_structured_output,
)


def test_invalid_output_is_retried() -> None:
    call_count = 0

    def operation() -> str:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            raise RuntimeError("第一次输出损坏")

        return "valid"

    result = retry_structured_output(
        operation,
        attempts=2,
        label="测试任务",
    )

    assert result == "valid"
    assert call_count == 2


def test_retry_exhaustion_raises_error() -> None:
    def operation() -> str:
        raise RuntimeError("始终无效")

    with pytest.raises(
        ModelOutputError,
        match="连续 2 次",
    ):
        retry_structured_output(
            operation,
            attempts=2,
            label="测试任务",
        )