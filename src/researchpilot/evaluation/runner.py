"""Call the running API, score cases, and write reproducible run artifacts."""

import math
import statistics
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from researchpilot.evaluation.models import (
    CaseMetrics,
    CaseResult,
    EvalCase,
    RunSummary,
)
from researchpilot.evaluation.scorers import case_passed, score_response


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    """Nearest-rank percentile, also defined for a single observation."""

    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return round(ordered[rank - 1], 2)


def _metric_values(
    results: list[CaseResult],
    field_name: str,
) -> list[float]:
    values: list[float] = []
    for result in results:
        value = getattr(result.metrics, field_name)
        if isinstance(value, (bool, int, float)):
            values.append(float(value))
    return values


class EvaluationRunner:
    """Synchronous runner for a small, paid-API evaluation suite."""

    def __init__(
        self,
        base_url: str,
        qa_top_k: int = 10,
        analysis_top_k_per_document: int = 8,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.qa_top_k = qa_top_k
        self.analysis_top_k_per_document = analysis_top_k_per_document
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
        )

    def close(self) -> None:
        self.client.close()

    def check_health(self) -> dict[str, Any]:
        try:
            response = self.client.get("/api/v1/health")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(
                f"无法连接 ResearchPilot API：{self.base_url}。"
                "请确认 Docker 和 uvicorn 已启动。"
            ) from exc

    def _request_for_case(
        self,
        case: EvalCase,
    ) -> tuple[str, dict[str, Any]]:
        if case.endpoint == "qa":
            return "/api/v1/qa/ask", {
                "query": case.question,
                "document_ids": case.document_ids,
                "top_k": self.qa_top_k,
            }

        return "/api/v1/analysis/experiments", {
            "document_ids": case.document_ids,
            "focus": case.focus,
            "top_k_per_document": self.analysis_top_k_per_document,
        }

    def run_case(self, case: EvalCase) -> CaseResult:
        path, request_body = self._request_for_case(case)
        started = perf_counter()
        status_code: int | None = None

        try:
            response = self.client.post(path, json=request_body)
            status_code = response.status_code
            latency_ms = round((perf_counter() - started) * 1000, 2)

            try:
                response_data = response.json()
            except ValueError:
                response_data = {"raw_text": response.text}

            if not response.is_success:
                error_detail = response_data.get("detail", response_data)
                return CaseResult(
                    case_id=case.case_id,
                    endpoint=case.endpoint,
                    request_body=request_body,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    response=response_data,
                    error=f"HTTP {status_code}: {error_detail}",
                    metrics=CaseMetrics(request_ok=False),
                    passed=False,
                )

            metrics = score_response(case, response_data)
            return CaseResult(
                case_id=case.case_id,
                endpoint=case.endpoint,
                request_body=request_body,
                status_code=status_code,
                latency_ms=latency_ms,
                response=response_data,
                metrics=metrics,
                passed=case_passed(metrics),
            )

        except httpx.HTTPError as exc:
            latency_ms = round((perf_counter() - started) * 1000, 2)
            return CaseResult(
                case_id=case.case_id,
                endpoint=case.endpoint,
                request_body=request_body,
                status_code=status_code,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
                metrics=CaseMetrics(request_ok=False),
                passed=False,
            )

    def run(
        self,
        cases: list[EvalCase],
        runs_dir: Path,
    ) -> tuple[Path, RunSummary]:
        self.check_health()

        now = datetime.now().astimezone()
        run_id = now.strftime("%Y%m%d_%H%M%S")
        output_dir = runs_dir / run_id
        suffix = 1
        while output_dir.exists():
            output_dir = runs_dir / f"{run_id}_{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True)

        results: list[CaseResult] = []
        results_path = output_dir / "results.jsonl"

        with results_path.open("w", encoding="utf-8") as output_file:
            for index, case in enumerate(cases, start=1):
                print(f"[{index}/{len(cases)}] {case.case_id} ...", flush=True)
                result = self.run_case(case)
                results.append(result)
                output_file.write(result.model_dump_json(exclude_none=True) + "\n")
                output_file.flush()

                status = "PASS" if result.passed else "FAIL"
                print(
                    f"    {status} | HTTP {result.status_code or '-'} "
                    f"| {result.latency_ms:.0f} ms",
                    flush=True,
                )

        summary = self._summarise(run_id, now.isoformat(), results)
        (output_dir / "summary.json").write_text(
            summary.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (output_dir / "report.md").write_text(
            self._render_report(summary, results),
            encoding="utf-8",
        )

        return output_dir, summary

    def _summarise(
        self,
        run_id: str,
        created_at: str,
        results: list[CaseResult],
    ) -> RunSummary:
        request_success_count = sum(result.metrics.request_ok for result in results)
        passed_count = sum(result.passed for result in results)
        case_count = len(results)

        return RunSummary(
            run_id=run_id,
            created_at=created_at,
            base_url=self.base_url,
            qa_top_k=self.qa_top_k,
            analysis_top_k_per_document=(self.analysis_top_k_per_document),
            case_count=case_count,
            request_success_count=request_success_count,
            passed_count=passed_count,
            failed_count=case_count - passed_count,
            request_success_rate=(
                request_success_count / case_count if case_count else None
            ),
            pass_rate=passed_count / case_count if case_count else None,
            sufficiency_accuracy=_mean(
                _metric_values(results, "sufficient_evidence_correct")
            ),
            citation_validity=_mean(_metric_values(results, "citation_validity")),
            document_coverage=_mean(_metric_values(results, "document_coverage")),
            gold_page_recall=_mean(_metric_values(results, "gold_page_recall")),
            anchor_evidence_recall=_mean(
                _metric_values(results, "anchor_evidence_recall")
            ),
            structure_completeness=_mean(
                _metric_values(results, "structure_completeness")
            ),
            mean_latency_ms=_mean([result.latency_ms for result in results]),
            p50_latency_ms=_percentile([result.latency_ms for result in results], 0.50),
            p95_latency_ms=_percentile([result.latency_ms for result in results], 0.95),
        )

    @staticmethod
    def _render_report(
        summary: RunSummary,
        results: list[CaseResult],
    ) -> str:
        def percent(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.1%}"

        lines = [
            f"# ResearchPilot 测评报告：{summary.run_id}",
            "",
            "## 汇总",
            "",
            f"- 案例数：{summary.case_count}",
            f"- 请求成功率：{percent(summary.request_success_rate)}",
            f"- 案例通过率：{percent(summary.pass_rate)}",
            f"- 证据充分性准确率：{percent(summary.sufficiency_accuracy)}",
            f"- 引用有效率：{percent(summary.citation_validity)}",
            f"- 文档覆盖率：{percent(summary.document_coverage)}",
            f"- 标准证据页召回率：{percent(summary.gold_page_recall)}",
            f"- 锚点证据召回率：{percent(summary.anchor_evidence_recall)}",
            f"- 实验分析结构完整性：{percent(summary.structure_completeness)}",
            f"- 平均耗时：{summary.mean_latency_ms or 0:.0f} ms",
            (
                f"- P50 / P95：{summary.p50_latency_ms or 0:.0f} / "
                f"{summary.p95_latency_ms or 0:.0f} ms"
            ),
            "",
            "## 案例明细",
            "",
            "| case_id | 状态 | HTTP | 充分性 | 文档覆盖 | 页召回 | 引用有效 | 耗时 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for result in results:
            metrics = result.metrics
            lines.append(
                "| "
                + " | ".join(
                    [
                        result.case_id,
                        "PASS" if result.passed else "FAIL",
                        str(result.status_code or "-"),
                        (
                            "Y"
                            if metrics.sufficient_evidence_correct is True
                            else "N"
                            if metrics.sufficient_evidence_correct is False
                            else "-"
                        ),
                        percent(metrics.document_coverage),
                        percent(metrics.gold_page_recall),
                        percent(metrics.citation_validity),
                        f"{result.latency_ms:.0f} ms",
                    ]
                )
                + " |"
            )

        failures = [result for result in results if not result.passed]
        if failures:
            lines.extend(["", "## 失败诊断", ""])
            for result in failures:
                metrics = result.metrics
                reasons: list[str] = []
                if result.error:
                    reasons.append(result.error.replace("\n", " "))
                if metrics.sufficient_evidence_correct is False:
                    reasons.append("证据充分性判断与预期不一致")
                if metrics.invalid_citation_ids:
                    reasons.append(f"无效引用 {metrics.invalid_citation_ids}")
                if metrics.missing_document_ids:
                    reasons.append(f"缺少文档 {metrics.missing_document_ids}")
                if metrics.missed_gold_evidence_indexes:
                    reasons.append(
                        "未命中 gold_evidence 序号 "
                        f"{metrics.missed_gold_evidence_indexes}"
                    )
                if metrics.missing_structure_items:
                    reasons.append(f"结构缺失 {metrics.missing_structure_items}")
                lines.append(
                    f"- `{result.case_id}`："
                    + ("；".join(reasons) if reasons else "未通过质量门槛")
                )

        return "\n".join(lines) + "\n"
