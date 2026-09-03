"""批量评测已有运行结果的语义质量。"""

import argparse
import json
from pathlib import Path
from typing import Any

from researchpilot.evaluation.loader import (
    load_suite,
)
from researchpilot.evaluation.semantic_judge import (
    SemanticJudge,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量执行语义质量评测。"
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="已有测评运行目录。",
    )

    parser.add_argument(
        "--case-id",
        action="append",
        help=(
            "只评测指定案例；"
            "可以重复填写多次。"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="只评测筛选后的前 N 条案例。",
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )

    items: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行"
                    "不是合法 JSON"
                ) from exc

            items.append(item)

    return items


def ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def format_percent(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.1%}"


def group_summary(
    items: list[dict[str, Any]],
    answerability: str,
) -> dict[str, Any]:
    group = [
        item
        for item in items
        if (
            item.get("status") == "judged"
            and item.get("answerability")
            == answerability
        )
    ]

    passed = sum(
        bool(
            item["metrics"][
                "semantic_passed"
            ]
        )
        for item in group
    )

    return {
        "case_count": len(group),
        "passed_count": passed,
        "pass_rate": ratio(
            passed,
            len(group),
        ),
    }


def build_summary(
    run_dir: Path,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    judged = [
        item
        for item in items
        if item.get("status") == "judged"
    ]

    gold_verdicts = [
        verdict
        for item in judged
        for verdict in (
            item["judgment"]["gold_points"]
        )
    ]

    limitation_verdicts = [
        verdict
        for item in judged
        for verdict in (
            item["judgment"][
                "expected_limitations"
            ]
        )
    ]

    forbidden_verdicts = [
        verdict
        for item in judged
        for verdict in (
            item["judgment"][
                "forbidden_claims"
            ]
        )
    ]

    passed_count = sum(
        bool(
            item["metrics"]["semantic_passed"]
        )
        for item in judged
    )

    gold_covered = sum(
        bool(item["satisfied"])
        for item in gold_verdicts
    )

    limitations_covered = sum(
        bool(item["satisfied"])
        for item in limitation_verdicts
    )

    forbidden_violations = sum(
        bool(item["violated"])
        for item in forbidden_verdicts
    )

    forbidden_safety = (
        1
        - forbidden_violations
        / len(forbidden_verdicts)
        if forbidden_verdicts
        else 1.0
    )

    return {
        "source_run_dir": str(run_dir),
        "case_count": len(items),
        "judged_count": len(judged),
        "skipped_count": sum(
            item.get("status") == "skipped"
            for item in items
        ),
        "judge_failed_count": sum(
            item.get("status")
            == "judge_failed"
            for item in items
        ),
        "semantic_passed_count": (
            passed_count
        ),
        "semantic_pass_rate": ratio(
            passed_count,
            len(judged),
        ),
        "gold_point_count": len(
            gold_verdicts
        ),
        "gold_point_coverage": ratio(
            gold_covered,
            len(gold_verdicts),
        ),
        "expected_limitation_count": len(
            limitation_verdicts
        ),
        "limitation_coverage": ratio(
            limitations_covered,
            len(limitation_verdicts),
        ),
        "forbidden_claim_count": len(
            forbidden_verdicts
        ),
        "forbidden_violation_count": (
            forbidden_violations
        ),
        "forbidden_claim_safety": (
            forbidden_safety
        ),
        "by_answerability": {
            answerability: group_summary(
                judged,
                answerability,
            )
            for answerability in (
                "full",
                "partial",
                "none",
            )
        },
    }


def build_report(
    summary: dict[str, Any],
    items: list[dict[str, Any]],
) -> str:
    lines = [
        "# ResearchPilot 批量语义测评报告",
        "",
        "## 汇总",
        "",
        (
            f"- 已评测案例："
            f"{summary['judged_count']}"
            f"/{summary['case_count']}"
        ),
        (
            "- 语义通过率："
            + format_percent(
                summary["semantic_pass_rate"]
            )
        ),
        (
            "- 标准事实覆盖率："
            + format_percent(
                summary["gold_point_coverage"]
            )
        ),
        (
            "- 必要限制覆盖率："
            + format_percent(
                summary["limitation_coverage"]
            )
        ),
        (
            "- 禁止结论安全率："
            + format_percent(
                summary[
                    "forbidden_claim_safety"
                ]
            )
        ),
        (
            "- Judge 失败数："
            f"{summary['judge_failed_count']}"
        ),
        "",
        "## 按可回答性划分",
        "",
        "| 类型 | 案例数 | 通过数 | 通过率 |",
        "|---|---:|---:|---:|",
    ]

    labels = {
        "full": "完整可回答",
        "partial": "部分可回答",
        "none": "不可回答",
    }

    for key, label in labels.items():
        group = (
            summary["by_answerability"][key]
        )

        lines.append(
            f"| {label} "
            f"| {group['case_count']} "
            f"| {group['passed_count']} "
            f"| {format_percent(group['pass_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 案例明细",
            "",
            (
                "| case_id | 类型 | 状态 "
                "| 事实覆盖 | 限制覆盖 "
                "| 禁止结论安全 |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    for item in items:
        if item.get("status") != "judged":
            lines.append(
                f"| {item['case_id']} "
                f"| {item.get('answerability', '-')} "
                f"| {item.get('status', '-')} "
                "| N/A | N/A | N/A |"
            )
            continue

        metrics = item["metrics"]

        lines.append(
            f"| {item['case_id']} "
            f"| {item['answerability']} "
            f"| {'PASS' if metrics['semantic_passed'] else 'FAIL'} "
            f"| {format_percent(metrics['gold_point_coverage'])} "
            f"| {format_percent(metrics['limitation_coverage'])} "
            f"| {format_percent(metrics['forbidden_claim_safety'])} |"
        )

    failed_items = [
        item
        for item in items
        if (
            item.get("status") != "judged"
            or not item["metrics"][
                "semantic_passed"
            ]
        )
    ]

    if failed_items:
        lines.extend(
            [
                "",
                "## 失败诊断",
                "",
            ]
        )

    for item in failed_items:
        case_id = item["case_id"]

        if item.get("status") != "judged":
            error = str(
                item.get("error", "未知错误")
            ).replace("\n", " ")

            lines.append(
                f"- `{case_id}`：{error}"
            )
            continue

        judgment = item["judgment"]
        reasons: list[str] = []

        for verdict in judgment["gold_points"]:
            if not verdict["satisfied"]:
                reasons.append(
                    "未覆盖 "
                    f"{verdict['criterion_id']}："
                    f"{verdict['reason']}"
                )

        for verdict in judgment[
            "expected_limitations"
        ]:
            if not verdict["satisfied"]:
                reasons.append(
                    "未说明 "
                    f"{verdict['criterion_id']}："
                    f"{verdict['reason']}"
                )

        for verdict in judgment[
            "forbidden_claims"
        ]:
            if verdict["violated"]:
                reasons.append(
                    "违反 "
                    f"{verdict['criterion_id']}："
                    f"{verdict['reason']}"
                )

        lines.append(
            f"- `{case_id}`："
            + "；".join(reasons)
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    if (
        args.limit is not None
        and args.limit < 1
    ):
        raise SystemExit(
            "--limit 必须大于或等于 1"
        )

    run_dir = args.run_dir

    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    source_results = load_jsonl(
        run_dir / "results.jsonl"
    )

    case_map = {
        case.case_id: case
        for case in load_suite(
            PROJECT_ROOT / "evals",
            "all",
        )
    }

    if args.case_id:
        selected_ids = set(args.case_id)

        source_results = [
            result
            for result in source_results
            if result.get("case_id")
            in selected_ids
        ]

        missing_ids = (
            selected_ids
            - {
                result.get("case_id")
                for result in source_results
            }
        )

        if missing_ids:
            raise SystemExit(
                "运行结果中没有这些案例："
                f"{sorted(missing_ids)}"
            )

    if args.limit is not None:
        source_results = (
            source_results[: args.limit]
        )

    output_path = (
        run_dir / "semantic_results.jsonl"
    )

    batch_items: list[dict[str, Any]] = []
    judge = SemanticJudge()

    try:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            for index, source_result in enumerate(
                source_results,
                start=1,
            ):
                case_id = str(
                    source_result.get("case_id")
                )

                print(
                    f"[{index}/{len(source_results)}] "
                    f"{case_id} ...",
                    flush=True,
                )

                case = case_map.get(case_id)

                if case is None:
                    item = {
                        "case_id": case_id,
                        "status": "skipped",
                        "error": "测评集中不存在该案例",
                    }

                elif not source_result.get(
                    "metrics", {}
                ).get("request_ok"):
                    item = {
                        "case_id": case_id,
                        "endpoint": case.endpoint,
                        "answerability": (
                            case.answerability
                        ),
                        "scope": case.scope,
                        "status": "skipped",
                        "error": "原始 API 请求失败",
                    }

                else:
                    response = source_result.get(
                        "response"
                    )

                    try:
                        if not isinstance(
                            response,
                            dict,
                        ):
                            raise ValueError(
                                "没有有效的 API 响应"
                            )

                        semantic_result = (
                            judge.judge(
                                case,
                                response,
                            )
                        )

                        item = (
                            semantic_result.model_dump(
                                mode="json"
                            )
                        )

                        item.update(
                            {
                                "endpoint": (
                                    case.endpoint
                                ),
                                "answerability": (
                                    case.answerability
                                ),
                                "scope": case.scope,
                                "status": "judged",
                            }
                        )

                    except Exception as exc:
                        item = {
                            "case_id": case_id,
                            "endpoint": case.endpoint,
                            "answerability": (
                                case.answerability
                            ),
                            "scope": case.scope,
                            "status": "judge_failed",
                            "error": (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                        }

                batch_items.append(item)

                output_file.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_file.flush()

                print(
                    f"    {item['status']}",
                    flush=True,
                )

    finally:
        judge.close()

    summary = build_summary(
        run_dir,
        batch_items,
    )

    (run_dir / "semantic_summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_dir / "semantic_report.md").write_text(
        build_report(
            summary,
            batch_items,
        ),
        encoding="utf-8",
    )

    print()
    print("批量语义测评完成。")
    print(
        "语义通过率："
        + format_percent(
            summary["semantic_pass_rate"]
        )
    )
    print(
        "标准事实覆盖率："
        + format_percent(
            summary["gold_point_coverage"]
        )
    )
    print(
        "必要限制覆盖率："
        + format_percent(
            summary["limitation_coverage"]
        )
    )
    print(
        "禁止结论安全率："
        + format_percent(
            summary[
                "forbidden_claim_safety"
            ]
        )
    )
    print(f"报告：{run_dir / 'semantic_report.md'}")


if __name__ == "__main__":
    main()