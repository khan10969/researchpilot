from fastapi.testclient import TestClient

from researchpilot.api.main import app


def test_request_has_observability_headers() -> None:
    client = TestClient(app)

    response = client.get(
        "/",
        headers={"X-Request-ID": "test-request-001"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-001"
    assert response.headers["Server-Timing"].startswith("app;dur=")


def test_invalid_request_id_is_replaced() -> None:
    client = TestClient(app)

    response = client.get(
        "/",
        headers={"X-Request-ID": "../invalid value"},
    )

    request_id = response.headers["X-Request-ID"]

    assert request_id != "../invalid value"
    assert len(request_id) == 32
