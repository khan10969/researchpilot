import os

import pytest

from researchpilot.storage.qdrant_store import (
    QdrantStore,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv(
            "RUN_INTEGRATION_TESTS"
        )
        != "1",
        reason=(
            "需要设置 "
            "RUN_INTEGRATION_TESTS=1"
        ),
    ),
]


def test_qdrant_collection_is_available() -> None:
    store = QdrantStore()

    assert store.client.collection_exists(
        store.collection_name
    )

    assert store.count_points() > 0