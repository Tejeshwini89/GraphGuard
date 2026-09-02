from __future__ import annotations

import pytest

from graphguard.neo4j_store import Neo4jStore


def test_neo4j_store_requires_password(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="NEO4J_PASSWORD must be set"):
        Neo4jStore()
