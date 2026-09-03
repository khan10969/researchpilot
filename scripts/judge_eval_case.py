"""对已有测评结果执行单案例语义评测。"""

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
        description="评测一条已有回答的语义质量。"
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="已有测评运行目录。",
    )

    parser.add_argument(
        "--case-id",
        required=True,
        help="需要评测的案例 ID。",
    )

    return parser.parse_args()


def load_result(
    path: Path,
    case_id: str,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            result = json.loads(line)

            if result.get("case_id") == case_id:
                return result

    raise ValueError(
        f"results.jsonl 中没有案例：{case_id}"
    )


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir

    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    cases = {
        case.case_id: case
        for case in load_suite(
            PROJECT_ROOT / "evals",
            "all",
        )
    }

    case = cases.get(args.case_id)

    if case is None:
        raise SystemExit(
            f"测评集中不存在：{args.case_id}"
        )

    result = load_result(
        run_dir / "results.jsonl",
        args.case_id,
    )

    if not result.get("metrics", {}).get(
        "request_ok"
    ):
        raise SystemExit(
            "该案例的原始 API 请求失败，"
            "不能进行语义评测。"
        )

    response = result.get("response")

    if not isinstance(response, dict):
        raise SystemExit(
            "该案例没有保存有效响应。"
        )

    judge = SemanticJudge()

    try:
        semantic_result = judge.judge(
            case,
            response,
        )
    finally:
        judge.close()

    output_path = (
        run_dir
        / f"semantic_{args.case_id}.json"
    )

    output_path.write_text(
        semantic_result.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    metrics = semantic_result.metrics

    print(f"案例：{args.case_id}")
    print(
        "标准事实覆盖率：",
        metrics.gold_point_coverage,
    )
    print(
        "限制说明覆盖率：",
        metrics.limitation_coverage,
    )
    print(
        "禁止结论安全率：",
        metrics.forbidden_claim_safety,
    )
    print(
        "语义通过：",
        metrics.semantic_passed,
    )
    print(f"结果文件：{output_path}")


if __name__ == "__main__":
    main()