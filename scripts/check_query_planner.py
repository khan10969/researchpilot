"""手动检查复杂问题的拆解结果。"""

import argparse

from researchpilot.rag.query_planner import (
    QueryPlanner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查复杂问题检索计划。"
    )

    parser.add_argument(
        "--question",
        required=True,
        help="需要拆解的问题。",
    )

    parser.add_argument(
        "--document-count",
        type=int,
        default=2,
        help="指定文献数量。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    planner = QueryPlanner()

    try:
        queries = planner.plan(
            query=args.question,
            document_count=(
                args.document_count
            ),
        )
    finally:
        planner.close()

    print()
    print("检索计划：")

    for index, query in enumerate(
        queries,
        start=1,
    ):
        label = (
            "原问题"
            if index == 1
            else f"子问题{index - 1}"
        )

        print(f"{label}：{query}")


if __name__ == "__main__":
    main()