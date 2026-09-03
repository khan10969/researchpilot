from openai import OpenAI

from researchpilot.config import settings


def main() -> None:
    if settings.llm_api_key is None:
        raise RuntimeError(
            "没有配置 LLM_API_KEY"
        )

    client = OpenAI(
        api_key=(
            settings.llm_api_key.get_secret_value()
        ),
        base_url=settings.llm_base_url,
        timeout=120.0,
        max_retries=2,
    )

    print(f"服务地址：{settings.llm_base_url}")
    print(f"模型：{settings.llm_model}")
    print("正在测试 DeepSeek API……")

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": "你是API连接测试助手。",
            },
            {
                "role": "user",
                "content": "只回复四个字：连接成功",
            },
        ],
        max_tokens=64,
        stream=False,
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )

    content = response.choices[0].message.content

    print(f"模型返回：{content}")


if __name__ == "__main__":
    main()