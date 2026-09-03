"""ResearchPilot Streamlit Web 界面。"""

import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_BASE_URL = os.getenv(
    "WEB_API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


def api_url(path: str) -> str:
    """构造后端接口地址。"""
    return f"{API_BASE_URL}/{path.lstrip('/')}"


def get_json(
    path: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """调用后端 GET 接口。"""
    response = httpx.get(
        api_url(path),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def post_json(
    path: str,
    payload: dict[str, Any],
    timeout: float = 300.0,
) -> dict[str, Any]:
    """调用后端 JSON POST 接口。"""
    response = httpx.post(
        api_url(path),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def upload_pdf(
    filename: str,
    content: bytes,
) -> dict[str, Any]:
    """向后端上传 PDF。"""
    response = httpx.post(
        api_url("/api/v1/documents/upload"),
        files={
            "file": (
                filename,
                content,
                "application/pdf",
            )
        },
        timeout=600.0,
    )
    response.raise_for_status()
    return response.json()


def format_error(exc: Exception) -> str:
    """提取适合显示给用户的错误信息。"""
    if isinstance(
        exc,
        httpx.HTTPStatusError,
    ):
        try:
            payload = exc.response.json()
            detail = payload.get("detail")

            if detail:
                return str(detail)
        except ValueError:
            pass

        return (
            f"后端返回状态码 "
            f"{exc.response.status_code}："
            f"{exc.response.text}"
        )

    if isinstance(
        exc,
        httpx.RequestError,
    ):
        return (
            "无法连接 ResearchPilot API。"
            "请确认 FastAPI 已启动。"
        )

    return str(exc)


def format_pages(
    page_start: int | None,
    page_end: int | None,
) -> str:
    """格式化证据页码。"""
    if page_start is None:
        return "页码未知"

    if (
        page_end is None
        or page_end == page_start
    ):
        return f"第 {page_start} 页"

    return f"第 {page_start}–{page_end} 页"


def render_qa_result(
    result: dict[str, Any],
) -> None:
    """显示问答结果和证据。"""
    answer = result["answer"]
    evidences = result.get(
        "evidences",
        [],
    )

    st.divider()
    st.subheader("回答")

    if answer["sufficient_evidence"]:
        st.success("当前文献证据足以支持回答")
    else:
        st.warning(
            "当前证据不足，以下内容包含可确认部分和限制说明"
        )

    st.markdown(answer["overview"])

    claims = answer.get("claims", [])

    if claims:
        st.markdown("#### 证据支持的结论")

        for index, claim in enumerate(
            claims,
            start=1,
        ):
            st.write(
                f"{index}. {claim['statement']}"
            )

            citation_text = "、".join(
                f"证据 {evidence_id}"
                for evidence_id
                in claim["evidence_ids"]
            )

            st.caption(
                f"引用：{citation_text}"
            )

    limitations = answer.get(
        "limitations",
        [],
    )

    if limitations:
        st.markdown("#### 证据边界与限制")

        for limitation in limitations:
            st.write(f"- {limitation}")

    retrieval_queries = result.get(
        "retrieval_queries",
        [],
    )

    if len(retrieval_queries) > 1:
        with st.expander(
            "查看系统使用的检索子问题"
        ):
            for query in retrieval_queries:
                st.write(f"- {query}")

    st.markdown("#### 可定位证据")

    for evidence in evidences:
        pages = format_pages(
            evidence.get("page_start"),
            evidence.get("page_end"),
        )

        title = (
            f"证据 {evidence['evidence_id']}｜"
            f"{evidence['source_file']}｜"
            f"{pages}"
        )

        with st.expander(title):
            st.caption(
                f"章节：{evidence['section']}"
            )

            score_text = (
                "向量检索分数："
                f"{evidence['score']:.4f}"
            )

            rerank_score = evidence.get(
                "rerank_score"
            )

            if rerank_score is not None:
                score_text += (
                    "　重排序分数："
                    f"{rerank_score:.4f}"
                )

            st.caption(score_text)
            st.write(evidence["text"])

            source_url = api_url(
                "/api/v1/documents/"
                f"{evidence['document_id']}"
                "/source"
            )

            page_start = evidence.get(
                "page_start"
            )

            if page_start is not None:
                source_url += (
                    f"#page={page_start}"
                )

            st.link_button(
                "打开原始 PDF",
                source_url,
            )


def load_health(
) -> tuple[
    dict[str, Any] | None,
    str | None,
]:
    """获取后端健康状态。"""
    try:
        return (
            get_json(
                "/api/v1/health",
                timeout=5.0,
            ),
            None,
        )
    except Exception as exc:
        return None, format_error(exc)


def load_documents(
) -> tuple[
    list[dict[str, Any]],
    str | None,
]:
    """读取文献列表。"""
    try:
        payload = get_json(
            "/api/v1/documents"
        )
        return payload.get("items", []), None
    except Exception as exc:
        return [], format_error(exc)


def main() -> None:
    st.set_page_config(
        page_title="ResearchPilot",
        page_icon="🧭",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1320px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e3e8f0;
            border-radius: 12px;
            padding: 12px 16px;
        }

        [data-testid="stExpander"] {
            background: #ffffff;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("ResearchPilot")
    st.caption(
        "面向科研文献与实验结果的证据驱动助手"
    )

    flash_message = st.session_state.pop(
        "flash_message",
        None,
    )

    if flash_message:
        st.success(flash_message)

    health, health_error = load_health()

    with st.sidebar:
        st.header("系统状态")

        if health is not None:
            st.success("API 与 Qdrant 已连接")
            st.caption(
                f"模型：{health['llm_model']}"
            )
            st.caption(
                "向量集合："
                f"{health['collection_name']}"
            )
            st.caption(
                f"索引点数：{health['point_count']}"
            )
        else:
            st.error("后端服务未连接")
            st.caption(health_error)

        st.divider()
        st.caption(
            f"API：{API_BASE_URL}"
        )

    documents: list[dict[str, Any]] = []
    documents_error: str | None = None

    if health is not None:
        documents, documents_error = (
            load_documents()
        )

    metric_columns = st.columns(3)

    metric_columns[0].metric(
        "已索引文献",
        len(documents),
    )

    metric_columns[1].metric(
        "向量数据",
        (
            health["point_count"]
            if health is not None
            else "不可用"
        ),
    )

    metric_columns[2].metric(
        "服务状态",
        (
            "正常"
            if health is not None
            else "未连接"
        ),
    )

    library_tab, qa_tab = st.tabs(
        [
            "文献库",
            "证据问答",
        ]
    )

    with library_tab:
        left_column, right_column = (
            st.columns([1, 1.6])
        )

        with left_column:
            st.subheader("上传文献")
            st.caption(
                "当前支持最大 50 MB 的 PDF 文件。"
            )

            uploaded_file = st.file_uploader(
                "选择公开论文或技术报告",
                type=["pdf"],
            )

            upload_clicked = st.button(
                "上传并建立索引",
                type="primary",
                disabled=(
                    uploaded_file is None
                    or health is None
                ),
            )

            if (
                upload_clicked
                and uploaded_file is not None
            ):
                try:
                    with st.spinner(
                        "正在解析、分块并建立索引……"
                    ):
                        upload_result = upload_pdf(
                            filename=(
                                uploaded_file.name
                            ),
                            content=(
                                uploaded_file.getvalue()
                            ),
                        )

                    duplicate_text = (
                        "，系统识别为重复文献"
                        if upload_result.get(
                            "duplicate"
                        )
                        else ""
                    )

                    st.session_state[
                        "flash_message"
                    ] = (
                        f"已索引 "
                        f"{upload_result['filename']}，"
                        f"生成 "
                        f"{upload_result['chunk_count']} "
                        f"个文本块"
                        f"{duplicate_text}。"
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(format_error(exc))

        with right_column:
            st.subheader("已索引文献")

            if documents_error:
                st.error(documents_error)
            elif not documents:
                st.info(
                    "当前还没有文献，请先上传 PDF。"
                )
            else:
                table_rows = [
                    {
                        "文件名": document[
                            "filename"
                        ],
                        "页数": document[
                            "page_count"
                        ],
                        "文本块": document[
                            "chunk_count"
                        ],
                        "文献ID": document[
                            "document_id"
                        ],
                    }
                    for document in documents
                ]

                st.dataframe(
                    table_rows,
                    use_container_width=True,
                    hide_index=True,
                )

    with qa_tab:
        st.subheader("基于文献证据提问")
        st.caption(
            "可以限定一篇或多篇文献；"
            "不选择时将在全部文献中检索。"
        )

        document_by_label = {
            (
                f"{document['filename']} · "
                f"{document['document_id'][-8:]}"
            ): document
            for document in documents
        }

        with st.form("qa_form"):
            selected_labels = st.multiselect(
                "限定检索文献",
                options=list(
                    document_by_label.keys()
                ),
            )

            question = st.text_area(
                "问题",
                placeholder=(
                    "例如：比较两篇论文的"
                    "实验设置、评价指标与局限性。"
                ),
                height=120,
            )

            top_k = st.slider(
                "基础召回证据数",
                min_value=4,
                max_value=20,
                value=10,
            )

            ask_clicked = st.form_submit_button(
                "检索证据并生成回答",
                type="primary",
                disabled=health is None,
            )

        if ask_clicked:
            cleaned_question = question.strip()

            if len(cleaned_question) < 2:
                st.warning(
                    "请输入至少两个字符的问题。"
                )
            else:
                document_ids = [
                    document_by_label[label][
                        "document_id"
                    ]
                    for label in selected_labels
                ]

                payload: dict[str, Any] = {
                    "query": cleaned_question,
                    "top_k": top_k,
                    "document_ids": (
                        document_ids or None
                    ),
                }

                try:
                    with st.spinner(
                        "正在规划检索、重排证据"
                        "并生成回答……"
                    ):
                        result = post_json(
                            "/api/v1/qa/ask",
                            payload=payload,
                            timeout=300.0,
                        )

                    st.session_state[
                        "last_qa_result"
                    ] = result

                except Exception as exc:
                    st.error(format_error(exc))

        last_result = st.session_state.get(
            "last_qa_result"
        )

        if last_result is not None:
            render_qa_result(last_result)


if __name__ == "__main__":
    main()