import argparse

from researchpilot.rag.service import (
    RAGService,
    render_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResearchPilot 文献证据问答"
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="需要回答的科研问题",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="召回的证据数量",
    )

    args = parser.parse_args()

    query = args.query

    if not query:
        query = input("请输入问题：").strip()

    if not query:
        raise ValueError("问题不能为空")

    service = RAGService()

    print("正在检索证据并生成回答……")

    result = service.answer(
        query=query,
        top_k=args.top_k,
    )

    print()
    print(render_result(result))


if __name__ == "__main__":
    main()