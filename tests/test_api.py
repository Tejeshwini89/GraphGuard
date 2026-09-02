from __future__ import annotations

from graphguard.api import EXPECTED_FEATURES, app


def test_api_contract() -> None:
    routes = {route.path for route in app.routes}
    assert {"/health", "/model", "/predict", "/explain"}.issubset(routes)
    assert EXPECTED_FEATURES == 165


def test_explain_route_is_post() -> None:
    explain_route = next(route for route in app.routes if route.path == "/explain")
    assert explain_route.methods == {"POST"}
