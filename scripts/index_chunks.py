import json
from pathlib import Path

from researchpilot.ingestion.models import DocumentChunk
from researchpilot.retrieval.embeddings import EmbeddingService
from researchpilot.storage.qdrant_store import QdrantStore

PROJECT_ROOT = Path(__file__).parents[1]
CHUNKS_DIR = PROJECT_ROOT / 'data' / 'chunks'


def load_chunks() -> list[DocumentChunk]:
    chunk_files = sorted(
        CHUNKS_DIR.glob('*.chunks.json'),
    )

    if not chunk_files:
        raise FileNotFoundError(
            f"在{CHUNKS_DIR}中没有找到 .chunks.json文件"
        )

    chunks_by_id: dict[str, DocumentChunk] = {}

    for path in chunk_files:
        data = json.loads(
            path.read_text(encoding='utf-8'),
        )

        for item in data:
            chunk = DocumentChunk.model_validate(item)
            chunks_by_id[chunk.chunk_id] = chunk

        print(f"已读取：{path.name}")

    return list(chunks_by_id.values())


def main() -> None:
    chunks = load_chunks()
    print(f"文本块总数：{len(chunks)}")

    embedding_service = EmbeddingService()

    # 将章节标题与正文一起嵌入，可以增强章节语义
    embedding_texts = [
        f"章节：{chunk.section}\n{chunk.text}"
        for chunk in chunks
        ]

    print("开始生成文档向量……")

    embeddings = embedding_service.encode_documents(embedding_texts)

    print(f"向量矩阵形状：{embeddings.shape}")

    store = QdrantStore()
    store.ensure_collection(
        vector_size=embedding_service.dimension
    )

    print("开始写入 Qdrant……")

    store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    point_count = store.count_points()

    print()
    print("索引构建完成！")
    print(f"Collection：{store.collection_name}")
    print(f"Qdrant 中的 Point 数量：{point_count}")


if __name__ == "__main__":
    main()