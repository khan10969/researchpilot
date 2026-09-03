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


def delete_json(
    path: str,
    params: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """调用后端 DELETE 接口。"""
    response = httpx.delete(
        api_url(path),
        params=params,
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

        return f"后端返回状态码 {exc.response.status_code}：{exc.response.text}"

    if isinstance(
        exc,
        httpx.RequestError,
    ):
        return "无法连接 ResearchPilot API。请确认 FastAPI 已启动。"

    return str(exc)


def format_pages(
    page_start: int | None,
    page_end: int | None,
) -> str:
    """格式化证据页码。"""
    if page_start is None:
        return "页码未知"

    if page_end is None or page_end == page_start:
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
        st.warning("当前证据不足，以下内容包含可确认部分和限制说明")

    st.markdown(answer["overview"])

    claims = answer.get("claims", [])

    if claims:
        st.markdown("#### 证据支持的结论")

        for index, claim in enumerate(
            claims,
            start=1,
        ):
            st.write(f"{index}. {claim['statement']}")

            citation_text = "、".join(
                f"证据 {evidence_id}" for evidence_id in claim["evidence_ids"]
            )

            st.caption(f"引用：{citation_text}")

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
        with st.expander("查看系统使用的检索子问题"):
            for query in retrieval_queries:
                st.write(f"- {query}")

    st.markdown("#### 可定位证据")

    for evidence in evidences:
        pages = format_pages(
            evidence.get("page_start"),
            evidence.get("page_end"),
        )

        title = f"证据 {evidence['evidence_id']}｜{evidence['source_file']}｜{pages}"

        with st.expander(title):
            st.caption(f"章节：{evidence['section']}")

            score_text = f"向量检索分数：{evidence['score']:.4f}"

            rerank_score = evidence.get("rerank_score")

            if rerank_score is not None:
                score_text += f"　重排序分数：{rerank_score:.4f}"

            st.caption(score_text)
            st.write(evidence["text"])

            source_url = api_url(f"/api/v1/documents/{evidence['document_id']}/source")

            page_start = evidence.get("page_start")

            if page_start is not None:
                source_url += f"#page={page_start}"

            st.link_button(
                "打开原始 PDF",
                source_url,
            )


def format_evidence_ids(
    evidence_ids: list[int],
) -> str:
    """格式化实验分析中的证据编号。"""
    if not evidence_ids:
        return "未绑定直接证据"

    return "、".join(f"证据 {evidence_id}" for evidence_id in evidence_ids)


def render_claim_group(
    title: str,
    claims: list[dict[str, Any]],
) -> None:
    """显示一组带引用的实验事实。"""
    if not claims:
        return

    st.markdown(f"**{title}**")

    for claim in claims:
        st.write(f"- {claim['statement']}")
        st.caption("引用：" + format_evidence_ids(claim.get("evidence_ids", [])))


def render_experiment_result(
    result: dict[str, Any],
) -> None:
    """显示实验分析结果。"""
    analysis = result["analysis"]

    st.divider()
    st.subheader("实验分析结果")

    if analysis["sufficient_evidence"]:
        st.success("当前文献证据足以支持本次分析")
    else:
        st.warning("当前证据不完全充分，请结合下方限制说明理解结论。")

    st.markdown("#### 总体结论")
    st.write(analysis.get("overall_conclusion") or "当前没有形成总体结论。")

    overall_evidence_ids = analysis.get(
        "overall_evidence_ids",
        [],
    )

    if overall_evidence_ids:
        st.caption("总体结论引用：" + format_evidence_ids(overall_evidence_ids))

    comparisons = analysis.get(
        "comparisons",
        [],
    )

    st.markdown("#### 跨文献比较")

    if comparisons:
        for comparison in comparisons:
            st.markdown(f"**{comparison['aspect']}**")
            st.write(comparison["finding"])
            st.caption(
                "引用："
                + format_evidence_ids(
                    comparison.get(
                        "evidence_ids",
                        [],
                    )
                )
            )
    else:
        st.info("当前没有跨文献比较结果。分析单篇文献时这是正常现象。")

    summaries = analysis.get(
        "document_summaries",
        [],
    )

    st.markdown("#### 分文献实验摘要")

    if not summaries:
        st.info("当前没有生成分文献摘要。")

    group_fields = [
        ("研究任务", "research_tasks"),
        ("方法", "methods"),
        ("数据集", "datasets"),
        ("基线方法", "baselines"),
        ("实验设置", "experiment_settings"),
        ("评价指标", "metrics"),
        ("实验结果", "results"),
    ]

    for summary in summaries:
        title = f"{summary['source_file']} · {summary['document_id'][-8:]}"

        with st.expander(
            title,
            expanded=True,
        ):
            for group_title, field_name in group_fields:
                render_claim_group(
                    group_title,
                    summary.get(field_name, []),
                )

            document_limitations = summary.get(
                "limitations",
                [],
            )

            if document_limitations:
                st.markdown("**文献限制**")

                for limitation in document_limitations:
                    st.write(f"- {limitation['statement']}")

                    evidence_ids = limitation.get(
                        "evidence_ids",
                        [],
                    )

                    if evidence_ids:
                        st.caption("引用：" + format_evidence_ids(evidence_ids))
                    else:
                        st.caption("该项是根据当前证据中缺少相关信息得出的限制。")

    limitations = analysis.get(
        "limitations",
        [],
    )

    st.markdown("#### 整体证据边界与限制")

    if limitations:
        for limitation in limitations:
            st.write(f"- {limitation}")
    else:
        st.write("未报告额外限制。")

    evidences = result.get(
        "evidences",
        [],
    )

    st.markdown("#### 可定位证据")

    if not evidences:
        st.info("本次分析没有检索到证据。")
        return

    for evidence in evidences:
        pages = format_pages(
            evidence.get("page_start"),
            evidence.get("page_end"),
        )

        evidence_id = evidence["evidence_id"]
        title = (
            f"证据 {evidence_id}｜"
            f"{evidence['source_file']}｜"
            f"{pages}｜"
            f"{evidence['evidence_type']}"
        )

        with st.expander(title):
            st.caption(f"章节：{evidence['section']}")

            retrieval_score = evidence.get("retrieval_score")

            if retrieval_score is not None:
                st.caption(f"检索分数：{retrieval_score:.4f}")

            st.write(evidence["text"])

            source_url = api_url(f"/api/v1/documents/{evidence['document_id']}/source")

            page_start = evidence.get("page_start")

            if page_start is not None:
                source_url += f"#page={page_start}"

            st.link_button(
                "打开原始 PDF",
                source_url,
                key=(f"analysis_evidence_{evidence_id}_{evidence['document_id']}"),
            )


def render_document_tables(
    payload: dict[str, Any],
) -> None:
    """显示从文献中提取的结构化表格。"""
    tables = payload.get("tables", [])

    st.caption(f"共提取到 {payload.get('table_count', 0)} 个表格")

    if not tables:
        st.info("这篇文献没有提取到结构化表格。")
        return

    for index, table in enumerate(
        tables,
        start=1,
    ):
        caption = table.get("caption") or table.get("source_ref") or f"表格 {index}"

        page_no = table.get("page_no")
        page_text = f"第 {page_no} 页" if page_no is not None else "页码未知"

        with st.expander(f"表格 {index}｜{page_text}｜{caption}"):
            st.caption(f"{table['num_rows']} 行 × {table['num_cols']} 列")

            rows = table.get("rows", [])

            if rows:
                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("系统记录了该表格的位置，但没有可显示的单元格内容。")

            source_url = api_url(f"/api/v1/documents/{table['document_id']}/source")

            if page_no is not None:
                source_url += f"#page={page_no}"

            st.link_button(
                "打开原始 PDF",
                source_url,
                key=f"table_{table['table_id']}",
            )


def load_health() -> tuple[
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


def load_documents() -> tuple[
    list[dict[str, Any]],
    str | None,
]:
    """读取文献列表。"""
    try:
        payload = get_json("/api/v1/documents")
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
    st.caption("面向科研文献与实验结果的证据驱动助手")

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
            st.caption(f"模型：{health['llm_model']}")
            st.caption(f"向量集合：{health['collection_name']}")
            st.caption(f"索引点数：{health['point_count']}")
        else:
            st.error("后端服务未连接")
            st.caption(health_error)

        st.divider()
        st.caption(f"API：{API_BASE_URL}")

    documents: list[dict[str, Any]] = []
    documents_error: str | None = None

    if health is not None:
        documents, documents_error = load_documents()

    metric_columns = st.columns(3)

    metric_columns[0].metric(
        "已索引文献",
        len(documents),
    )

    metric_columns[1].metric(
        "向量数据",
        (health["point_count"] if health is not None else "不可用"),
    )

    metric_columns[2].metric(
        "服务状态",
        ("正常" if health is not None else "未连接"),
    )

    document_by_label = {
        (f"{document['filename']} · {document['document_id'][-8:]}"): document
        for document in documents
    }

    library_tab, qa_tab, analysis_tab = st.tabs(
        [
            "文献库",
            "证据问答",
            "实验分析",
        ]
    )

    with library_tab:
        left_column, right_column = st.columns([1, 1.6])

        with left_column:
            st.subheader("上传文献")
            st.caption("当前支持最大 50 MB 的 PDF 文件。")

            uploaded_file = st.file_uploader(
                "选择公开论文或技术报告",
                type=["pdf"],
            )

            upload_clicked = st.button(
                "上传并建立索引",
                type="primary",
                disabled=(uploaded_file is None or health is None),
            )

            if upload_clicked and uploaded_file is not None:
                try:
                    with st.spinner("正在解析、分块并建立索引……"):
                        upload_result = upload_pdf(
                            filename=(uploaded_file.name),
                            content=(uploaded_file.getvalue()),
                        )

                    duplicate_text = (
                        "，系统识别为重复文献" if upload_result.get("duplicate") else ""
                    )

                    st.session_state["flash_message"] = (
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
                st.info("当前还没有文献，请先上传 PDF。")
            else:
                table_rows = [
                    {
                        "文件名": document["filename"],
                        "页数": document["page_count"],
                        "文本块": document["chunk_count"],
                        "文献ID": document["document_id"],
                    }
                    for document in documents
                ]

                st.dataframe(
                    table_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()
                st.markdown("#### 文献详情与管理")

                selected_library_label = st.selectbox(
                    "选择一篇文献",
                    options=list(document_by_label.keys()),
                )

                selected_document = document_by_label[selected_library_label]

                detail_columns = st.columns(3)

                detail_columns[0].metric(
                    "页数",
                    selected_document["page_count"],
                )

                detail_columns[1].metric(
                    "文本块",
                    selected_document["chunk_count"],
                )

                detail_columns[2].metric(
                    "索引状态",
                    selected_document["status"],
                )

                st.caption(f"文献 ID：{selected_document['document_id']}")
                st.caption(f"索引时间：{selected_document['indexed_at']}")
                st.caption(f"向量集合：{selected_document['collection_name']}")

                source_url = api_url(
                    f"/api/v1/documents/{selected_document['document_id']}/source"
                )

                st.link_button(
                    "打开原始 PDF",
                    source_url,
                    key=(f"library_source_{selected_document['document_id']}"),
                )

                with st.expander("删除这篇文献"):
                    st.warning(
                        "删除后，该文献的向量会从 "
                        "Qdrant 中移除，无法继续参与"
                        "检索和分析。原始文件会移动到 "
                        "data/trash，不会立即永久删除。"
                    )

                    st.write("请输入以下完整文件名确认：")
                    st.code(selected_document["filename"])

                    confirmation = st.text_input(
                        "确认文件名",
                        key=(f"delete_confirmation_{selected_document['document_id']}"),
                    )

                    understood = st.checkbox(
                        "我确认删除这篇文献及其向量索引",
                        key=(f"delete_understood_{selected_document['document_id']}"),
                    )

                    filename_matches = confirmation == selected_document["filename"]

                    if confirmation and not filename_matches:
                        st.caption("输入的文件名尚未完全匹配。")

                    delete_clicked = st.button(
                        "确认删除",
                        type="primary",
                        disabled=not (
                            filename_matches and understood and health is not None
                        ),
                        key=(f"delete_document_{selected_document['document_id']}"),
                    )

                    if delete_clicked:
                        try:
                            with st.spinner("正在删除向量并移动本地文献文件……"):
                                delete_result = delete_json(
                                    "/api/v1/documents/"
                                    f"{selected_document['document_id']}",
                                    params={"confirm": "true"},
                                )

                            for state_key in [
                                "last_qa_result",
                                "last_experiment_result",
                                "last_table_result",
                                "table_document",
                            ]:
                                st.session_state.pop(
                                    state_key,
                                    None,
                                )

                            st.session_state["flash_message"] = (
                                f"已移除 "
                                f"{delete_result['filename']}，"
                                f"删除 "
                                f"{delete_result['deleted_point_count']} "
                                f"条向量。原始文件已移动到 "
                                f"data/trash/"
                                f"{delete_result['trash_entry']}。"
                            )

                            st.rerun()

                        except Exception as exc:
                            st.error(format_error(exc))

    with qa_tab:
        st.subheader("基于文献证据提问")
        st.caption("可以限定一篇或多篇文献；不选择时将在全部文献中检索。")

        with st.form("qa_form"):
            selected_labels = st.multiselect(
                "限定检索文献",
                options=list(document_by_label.keys()),
            )

            question = st.text_area(
                "问题",
                placeholder=("例如：比较两篇论文的实验设置、评价指标与局限性。"),
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
                st.warning("请输入至少两个字符的问题。")
            else:
                document_ids = [
                    document_by_label[label]["document_id"] for label in selected_labels
                ]

                payload: dict[str, Any] = {
                    "query": cleaned_question,
                    "top_k": top_k,
                    "document_ids": (document_ids or None),
                }

                try:
                    with st.spinner("正在规划检索、重排证据并生成回答……"):
                        result = post_json(
                            "/api/v1/qa/ask",
                            payload=payload,
                            timeout=300.0,
                        )

                    st.session_state["last_qa_result"] = result

                except Exception as exc:
                    st.error(format_error(exc))

        last_result = st.session_state.get("last_qa_result")

        if last_result is not None:
            render_qa_result(last_result)

    with analysis_tab:
        st.subheader("实验结果分析")
        st.caption(
            "选择 1～5 篇文献，提取实验设置、"
            "数据集、指标、结果与限制；"
            "选择多篇文献时还会生成跨文献比较。"
        )

        if not documents:
            st.info("当前没有可分析的文献，请先在文献库上传并索引 PDF。")
        else:
            with st.form("experiment_analysis_form"):
                selected_analysis_labels = st.multiselect(
                    "选择需要分析的文献",
                    options=list(document_by_label.keys()),
                    max_selections=5,
                )

                analysis_focus = st.text_area(
                    "重点分析内容（可选）",
                    placeholder=(
                        "例如：重点比较数据集、噪声设置、评价指标和不同方法的性能差异。"
                    ),
                    height=100,
                )

                top_k_per_document = st.slider(
                    "每篇文献召回的正文证据数",
                    min_value=4,
                    max_value=15,
                    value=8,
                )

                analyze_clicked = st.form_submit_button(
                    "开始实验分析",
                    type="primary",
                    disabled=health is None,
                )

            if analyze_clicked:
                if not selected_analysis_labels:
                    st.warning("请至少选择一篇文献。")
                else:
                    document_ids = [
                        document_by_label[label]["document_id"]
                        for label in selected_analysis_labels
                    ]

                    payload = {
                        "document_ids": document_ids,
                        "focus": (analysis_focus.strip() or None),
                        "top_k_per_document": (top_k_per_document),
                    }

                    try:
                        with st.spinner("正在检索实验信息、读取表格并生成结构化分析……"):
                            result = post_json(
                                ("/api/v1/analysis/experiments"),
                                payload=payload,
                                timeout=600.0,
                            )

                        st.session_state["last_experiment_result"] = result

                    except Exception as exc:
                        st.error(format_error(exc))

            last_experiment_result = st.session_state.get("last_experiment_result")

            if last_experiment_result is not None:
                render_experiment_result(last_experiment_result)

            st.divider()
            st.subheader("原始结构化表格")
            st.caption("查看 Docling 从论文中提取出的表格，用于人工核对模型分析。")

            selected_table_label = st.selectbox(
                "选择一篇文献",
                options=list(document_by_label.keys()),
                key="table_document",
            )

            load_tables_clicked = st.button(
                "读取该文献的表格",
                key="load_document_tables",
            )

            if load_tables_clicked:
                selected_document = document_by_label[selected_table_label]

                try:
                    with st.spinner("正在读取结构化表格……"):
                        table_result = get_json(
                            "/api/v1/documents/"
                            f"{selected_document['document_id']}"
                            "/tables",
                            timeout=120.0,
                        )

                    st.session_state["last_table_result"] = table_result

                except Exception as exc:
                    st.error(format_error(exc))

            last_table_result = st.session_state.get("last_table_result")

            selected_document_id = document_by_label[selected_table_label][
                "document_id"
            ]

            if (
                last_table_result is not None
                and last_table_result["document_id"] == selected_document_id
            ):
                render_document_tables(last_table_result)


if __name__ == "__main__":
    main()
