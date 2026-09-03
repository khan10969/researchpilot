import json
from pathlib import Path

from docling.document_converter import DocumentConverter

# from torch.ao.quantization.fx import convert

#
# pdf_path = Path("../data/samples/HRRP_Radar_Target_Recognition_base_on_statistic_features_DongJian.pdf")
#
# if not pdf_path.exists():
#     raise FileNotFoundError(f"没有找到 PDF：{pdf_path.resolve()}")
#
# converter = DocumentConverter()
# result = converter.convert(pdf_path)
#
# markdown = result.document.export_to_markdown()
#
# print(f"PDF：{pdf_path}")
# print(f"解析字符数：{len(markdown)}")
# print()
# print("前 2000 个字符：")
# print(markdown[:2000])


# 获取项目根目录
# check_docling.py 位于项目根目录/scripts/下，因此parents[1]就是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    /"samples"
    /"HRRP_Radar_Target_Recognition_base_on_statistic_features_DongJian.pdf"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "artifacts"

def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"没有找到PDF： {PDF_PATH}")

    # if data/artifacts 不存在，则自动创建
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"开始解析 PDF： {PDF_PATH}")
    print("第一次运行可能需要下载Docling模型，请耐心等待.....")

    converter = DocumentConverter()
    result = converter.convert(PDF_PATH)

    document = result.document

    #导出为便于阅读的Markdown
    markdown = document.export_to_markdown()

    #导出为保留结构、页码和坐标信息的字典
    document_dict = document.export_to_dict()

    markdown_path = OUTPUT_DIR / f"{PDF_PATH.stem}.md"
    json_path = OUTPUT_DIR / f"{PDF_PATH.stem}.json"

    markdown_path.write_text(markdown,encoding="utf-8")

    json_path.write_text(
        json.dumps(
            document_dict,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("解析完成！")
    print(f"Markdown 字符数：{len(markdown)}")
    print(f"Markdown 输出：{markdown_path}")
    print(f"JSON 输出：{json_path}")

    print()
    print("解析结果前 1000 个字符：")
    print("-" * 60)
    print(markdown[:1000])
    print("-" * 60)

if __name__ == "__main__":
    main()
