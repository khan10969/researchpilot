import pytest
from fastapi.testclient import TestClient

from researchpilot.api.dependencies import (
    get_rag_service,
)
from researchpilot.api.main import app
from researchpilot.rag.models import (
    AnswerClaim,
    GroundedAnswer,
    RAGResult,
    RetrievedEvidence,
)


class FakeRAGService:
    """不会加载模型或调用 DeepSeek。"""

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> RAGResult:
        return RAGResult(
            query=query,
            document_ids=document_ids,
            answer=GroundedAnswer(
                sufficient_evidence=True,
                overview="这是测试回答。",
                overview_evidence_ids=[1],
                claims=[
                    AnswerClaim(
                        statement="这是测试结论。",
                        evidence_ids=[1],
                    )
                ],
                limitations=[],
            ),
            evidences=[
                RetrievedEvidence(
                    evidence_id=1,
                    score=0.9,
                    document_id="doc-1",
                    chunk_id="test-chunk",
                    source_file="test.pdf",
                    section="摘要",
                    page_start=1,
                    page_end=1,
                    text="这是测试证据。",
                    source_locations=[],
                )
            ],
        )


@pytest.fixture(autouse=True)
def override_dependencies():
    app.dependency_overrides[
        get_rag_service
    ] = lambda: FakeRAGService()

    yield

    app.dependency_overrides.clear()


def test_ask_endpoint() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/qa/ask",
        json={
            "query": "论文提出了什么方法？",
            "top_k": 5,
            "document_ids": ["doc-1"],
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == (
        "论文提出了什么方法？"
    )

    assert (
        body["answer"]["overview"]
        == "这是测试回答。"
    )

    assert (
        body["answer"]
        ["overview_evidence_ids"]
        == [1]
    )


def test_invalid_top_k_returns_422() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/qa/ask",
        json={
            "query": "测试问题",
            "top_k": 100,
        },
    )

    assert response.status_code == 422
