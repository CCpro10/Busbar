"""Exercise the public HTTP lifecycle rather than individual endpoint implementations."""

from fastapi.testclient import TestClient

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
