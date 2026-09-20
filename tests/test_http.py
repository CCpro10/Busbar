"""Exercise the public HTTP lifecycle rather than individual endpoint implementations."""

from importlib.metadata import version

import pytest
from fastapi.testclient import TestClient

from busbar.backends.base import BackendResult
from busbar.server import create_app


def test_http_lifecycle(runtime, questions):
    """A caller can create, reuse, version, inspect, decide, filter and remove contexts."""
    with TestClient(create_app(runtime)) as client:
        assert client.get("/health").json()["status"] == "ready"
        assert client.get("/v1/contexts").json() == []
        payload = {"namespace": "demo", "state": {"unused": True}}
        created = client.post("/v1/contexts", json=payload)
        assert created.status_code == 200
        key = created.json()["snapshot"]["id"]
        assert client.post("/v1/contexts", json=payload).json()["reused"]
        request = {"snapshot_id": key, "namespace": "demo", "questions": questions}
        response = client.post("/v1/decisions", json=request)
        assert response.status_code == 200
        assert set(response.json()["decisions"]) == set(questions)
        assert client.get(f"/v1/contexts/{key}").status_code == 404
        assert client.get(f"/v1/contexts/{key}?namespace=demo").status_code == 200
        assert len(client.get(f"/v1/contexts?namespace=demo&id_prefix={key[:8]}").json()) == 1
        updated = client.put(f"/v1/contexts/{key}", json={**payload, "state": {"unused": False}})
        assert updated.status_code == 200 and updated.json()["snapshot"]["id"] != key
        assert client.post("/v1/decisions", json={**request, "questions": {}}).status_code == 422
        assert client.delete(f"/v1/contexts/{key}?namespace=demo").json()["deleted"]
        assert not client.delete(f"/v1/contexts/{key}?namespace=demo").json()["deleted"]
        assert client.post("/v1/decisions", json=request).status_code == 404
        assert client.delete("/v1/contexts?namespace=demo").json()["deleted_count"] == 1
        assert client.get("/v1/stats").json()["cache_bytes"] == 0


def test_http_input_errors_do_not_create_state(runtime):
    """Unknown fields, invalid JSON and overlong contexts cannot create partial snapshots."""
    with TestClient(create_app(runtime)) as client:
        assert client.post("/v1/contexts", json={"state": {}, "typo": 1}).status_code == 422
        assert client.post("/v1/contexts", content="{").status_code == 422
        assert client.post("/v1/contexts", json={"state": "x" * 9000}).status_code == 422
        assert runtime.stats()["contexts"] == 0


@pytest.mark.parametrize(
    "bad",
    [
        BackendResult([[float("nan"), 0]], 1, 1),
        BackendResult([[0]], 1, 1),
        BackendResult([[0, 1]], -1, 1),
        BackendResult([[0, 1]], 1, 1, details={"bad": float("inf")}),
        BackendResult([[0, 1]], 1, 1, details={"logit_space": 7}),
        BackendResult([[0, 1]], 1, 1, details={"probability_status": None}),
    ],
)
def test_bad_backend_output_is_not_reported_as_bad_input(runtime, backend, monkeypatch, bad):
    """A failed scorer returns 502; callers can reuse the intact snapshot after recovery."""
    with TestClient(create_app(runtime)) as client:
        key = client.post("/v1/contexts", json={"state": "x"}).json()["snapshot"]["id"]
        payload = {"snapshot_id": key, "questions": {"q": {"type": "boolean", "question": "Q"}}}
        with monkeypatch.context() as patch:
            patch.setattr(backend, "score", lambda *args: bad)
            response = client.post("/v1/decisions", json=payload)
        assert response.status_code == 502
        assert response.json()["detail"] == "backend returned invalid decision output"
        assert client.post("/v1/decisions", json=payload).status_code == 200


def test_whitespace_candidate_is_rejected_before_inference(runtime, backend):
    """Both ordinary callers and training data need meaningful candidate descriptions."""
    with TestClient(create_app(runtime)) as client:
        key = client.post("/v1/contexts", json={"state": "x"}).json()["snapshot"]["id"]
        question = {
            "type": "choice",
            "question": "Q",
            "options": [{"id": "a", "description": "Yes"}, {"id": "b", "description": "  \n"}],
        }
        response = client.post(
            "/v1/decisions", json={"snapshot_id": key, "questions": {"q": question}}
        )
        assert response.status_code == 422 and not backend.forwards


def test_http_schema_version_matches_installed_distribution(runtime):
    """The public API must report the installed release rather than a stale second constant."""
    with TestClient(create_app(runtime)) as client:
        assert client.get("/openapi.json").json()["info"]["version"] == version("busbar-runtime")
