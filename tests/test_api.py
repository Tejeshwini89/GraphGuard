from __future__ import annotations

from graphguard.api import EXPECTED_FEATURES, app


def test_api_contract() -> None:
    routes = {route.path for route in app.routes}
    assert {"/health", "/model", "/predict"}.issubset(routes)
    assert EXPECTED_FEATURES == 165
