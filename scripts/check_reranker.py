from researchpilot.rag.reranker import (
    RerankerService,
)


def main() -> None:
    query = (
        "多球体FSOSR方法使用哪些"
        "评价指标衡量未知类识别？"
    )

    texts = [
        (
            "该方法使用ACC衡量已知类分类，"
            "使用AUROC和FPR95衡量"
            "未知类识别性能。"
        ),
        (
            "实验采用Adam优化器，"
            "初始学习率为0.001。"
        ),
        (
            "今天气温较高，适合进行户外活动。"
        ),
    ]

    service = RerankerService()

    results = service.rerank(
        query=query,
        texts=texts,
        top_k=len(texts),
    )

    print()
    print("重排序结果：")

    for rank, item in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"score={item.score:.6f} | "
            f"{item.text}"
        )


if __name__ == "__main__":
    main()