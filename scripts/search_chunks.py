import argparse

from researchpilot.retrieval.embeddings import EmbeddingService
from researchpilot.storage.qdrant_store import QdrantStore


def format_pages(
        page_start: int | None,
        page_end: int | None,
)->str:
    if page_start is None:
        return "未知页码"

    if page_end is None or page_start == page_end:
        return f"第{page_start} 页"

    return f"第{page_start}--{page_end}页"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 Qdrant 中检索论文证据",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="需要检索的问题",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="返回的证据数量",
    )

    args = parser.parse_args()

    query = args.query

    if not query:
        query = input("请输入问题：").strip()

    if not query:
        raise ValueError("问题不能为空")

    embedding_service = EmbeddingService()
    store = QdrantStore()

    if not store.client.collection_exists(
        store.collection_name
    ):
        raise RuntimeError(
            f"Collection {store.collection_name} 不存在，"
            "请先运行 index_chunks.py"
        )

    query_vector = embedding_service.encode_query(query)

    results = store.search(
        query_vector=query_vector,
        limit=args.limit,
    )

    print()
    print(f"问题：{query}")
    print(f"共返回 {len(results)} 条证据")
    print("=" * 70)


    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}

        pages = format_pages(
            payload.get("page_start"),
            payload.get("page_end"),
        )
        print()
        print(f"[证据 {rank}]")
        print(f"相似度：{result.score:.4f}")
        print(f"论文：{payload.get('source_file')}")
        print(f"章节：{payload.get('section')}")
        print(f"位置：{pages}")
        print(f"Chunk ID：{payload.get('chunk_id')}")
        print("-" * 70)
        print(payload.get("text", "")[:1200])
        print("-" * 70)



if __name__ == "__main__":
    main()