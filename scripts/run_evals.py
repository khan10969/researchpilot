"""Command-line entry point for the ResearchPilot v1 evaluation suite."""

import argparse
from pathlib import Path

from researchpilot.evaluation.loader import load_suite
from researchpilot.evaluation.runner import EvaluationRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 ResearchPilot 确定性测评并生成报告。"
    )
    parser.add_argument(
        "--suite",
        choices=("qa", "experiment", "all"),
        default="all",
        help="选择问答、实验分析或全部案例。",
    )
    # parser.add_argument(
    #     "--case-id",
    #     help="只运行指定 case_id。",
    # )
    parser.add_argument(
        "--case-id",
        action="append",
        help="只运行指定 case_id；可重复传入。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="只运行筛选后最前面的 N 条案例。",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="ResearchPilot API 地址。",
    )
    parser.add_argument("--qa-top-k", type=int, default=10)
    parser.add_argument("--analysis-top-k", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须大于或等于 1")
    if not 1 <= args.qa_top_k <= 20:
        raise SystemExit("--qa-top-k 必须在 1 到 20 之间")
    if not 4 <= args.analysis_top_k <= 15:
        raise SystemExit("--analysis-top-k 必须在 4 到 15 之间")

    cases = load_suite(PROJECT_ROOT / "evals", args.suite)

    # if args.case_id:
    #     cases = [case for case in cases if case.case_id == args.case_id]
    #     if not cases:
    #         raise SystemExit(f"没有找到 case_id：{args.case_id}")

    if args.case_id:
        selected_ids = set(args.case_id)
        cases = [
            case
            for case in cases
            if case.case_id in selected_ids
        ]

        found_ids = {case.case_id for case in cases}
        missing_ids = selected_ids - found_ids

        if missing_ids:
            missing_text = ", ".join(sorted(missing_ids))
            raise SystemExit(f"没有找到以下 case_id：{missing_text}")

    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"准备运行 {len(cases)} 条测评案例。")
    print("提示：每条案例通常会调用一次 DeepSeek API。")

    runner = EvaluationRunner(
        base_url=args.base_url,
        qa_top_k=args.qa_top_k,
        analysis_top_k_per_document=args.analysis_top_k,
        timeout_seconds=args.timeout,
    )

    try:
        output_dir, summary = runner.run(
            cases=cases,
            runs_dir=PROJECT_ROOT / "evals" / "runs",
        )
    finally:
        runner.close()

    print()
    print("测评完成。")
    print(f"通过：{summary.passed_count}/{summary.case_count}")
    print(f"报告目录：{output_dir}")


if __name__ == "__main__":
    main()
