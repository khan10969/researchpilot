import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from time import perf_counter
from typing import Any
from uuid import uuid4

import structlog

_REQUEST_ID: ContextVar[str] = ContextVar(
    "request_id",
    default="-",
)

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def configure_logging() -> None:
    """将应用日志配置为适合容器采集的 JSON。"""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(
                fmt="iso",
                utc=True,
            ),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        logger_factory=(structlog.PrintLoggerFactory(file=sys.stdout)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """创建结构化日志记录器。"""
    return structlog.get_logger(name)


def resolve_request_id(
    candidate: str | None,
) -> str:
    """验证外部请求编号，非法时重新生成。"""
    if candidate is not None:
        candidate = candidate.strip()

        if _REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate

    return uuid4().hex


def bind_request_id(
    request_id: str,
) -> Token:
    """把请求编号绑定到当前请求上下文。"""
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token) -> None:
    """请求结束后恢复上下文。"""
    _REQUEST_ID.reset(token)


def get_request_id() -> str:
    """取得当前请求编号。"""
    return _REQUEST_ID.get()


@contextmanager
def timed_stage(
    logger: Any,
    stage: str,
    **fields: Any,
) -> Iterator[None]:
    """记录一个处理阶段的耗时与结果。"""
    started = perf_counter()

    try:
        yield
    except Exception:
        duration_ms = round(
            (perf_counter() - started) * 1000,
            2,
        )

        logger.exception(
            "stage_failed",
            request_id=get_request_id(),
            stage=stage,
            duration_ms=duration_ms,
            **fields,
        )
        raise
    else:
        duration_ms = round(
            (perf_counter() - started) * 1000,
            2,
        )

        logger.info(
            "stage_completed",
            request_id=get_request_id(),
            stage=stage,
            duration_ms=duration_ms,
            **fields,
        )
