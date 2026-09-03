import argparse
import json
from pathlib import Path

from researchpilot.ingestion.chunker import build_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "chunks"


def find_default_input() -> Path:
    json_files = sorted(ARTIFACT_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"在 {ARTIFACT_DIR} 中没有找到 Docling JSON 文件"
        )

    return json_files[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Docling JSON 转换为检索文本块"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Docling JSON 文件路径",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="每个文本块的最大字符数",
    )

    args = parser.parse_args()

    input_path = args.input or find_default_input()
    input_path = input_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在：{input_path}")

    document = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    chunks = build_chunks(
        document,
        max_chars=args.max_chars,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / f"{input_path.stem}.chunks.json"
    )

    output_data = [
        chunk.model_dump()
        for chunk in chunks
    ]

    output_path.write_text(
        json.dumps(
            output_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    table_chunk_count = sum(
        "table" in chunk.element_types
        for chunk in chunks
    )

    print(f"输入文件：{input_path}")
    print(f"生成文本块：{len(chunks)}")
    print(f"包含表格的文本块：{table_chunk_count}")
    print(f"输出文件：{output_path}")

    if chunks:
        first = chunks[0]

        print()
        print("第一个文本块：")
        print(f"chunk_id：{first.chunk_id}")
        print(f"section：{first.section}")
        print(
            f"pages：{first.page_start} - {first.page_end}"
        )
        print("-" * 60)
        print(first.text[:1000])
        print("-" * 60)


if __name__ == "__main__":
    main()