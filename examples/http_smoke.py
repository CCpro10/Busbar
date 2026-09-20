"""Verify a running local Busbar server using actual TCP requests and its resident model."""

import argparse
import json
from pathlib import Path

import httpx


def exercise(base_url: str) -> dict:
    """Cover cross-request reuse, all decision types, fanout and snapshot removal."""
    example = json.loads(Path(__file__).with_name("returns.json").read_text())
    context = {**example["context"], "namespace": "http-smoke"}
    events = []
    with httpx.Client(base_url=base_url, timeout=120) as client:

        def call(method, path, payload=None, expected=200):
            """Record complete successful/error responses for an auditable integration run."""
            response = client.request(method, path, json=payload)
            assert response.status_code == expected, (method, path, response.text)
            body = response.json()
            events.append({"method": method, "path": path, "status": expected, "body": body})
            return body

        health = call("GET", "/health")
        call("DELETE", "/v1/contexts?namespace=http-smoke")
        compiled = call("POST", "/v1/contexts", context)
        key = compiled["snapshot"]["id"]
        reused = call("POST", "/v1/contexts", context)
        assert reused["reused"] and reused["prefill_tokens"] == 0
        request = {"namespace": "http-smoke", "snapshot_id": key, "questions": example["questions"]}
        decisions = call("POST", "/v1/decisions", request)
        assert set(decisions["decisions"]) == set(example["questions"])
        fanout = call(
            "POST",
            "/v1/decisions",
            {
                **request,
                "questions": {f"route{i:02}": example["questions"]["route"] for i in range(64)},
            },
        )
        assert len(fanout["decisions"]) == 64
        engine_managed = "engine_cached_tokens" in fanout["backend_details"]
        assert fanout["batches"] == (1 if engine_managed else 8)
        assert fanout["computed_tokens"] < fanout["reused_prefix_tokens"]
        repeated = call("POST", "/v1/decisions", request)
        for name, first in decisions["decisions"].items():
            second = repeated["decisions"][name]
            assert first["selected"] == second["selected"]
            # APC changes prefill shapes; low-precision kernels can differ slightly.
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(first["probabilities"], second["probabilities"], strict=True)
                )
                < 0.02
            )
        call("POST", "/v1/decisions", {**request, "questions": {}}, expected=422)
        if engine_managed:
            call("POST", "/v1/decisions", {**request, "projection": "selected"}, expected=422)
        call("POST", "/v1/decisions", {**request, "namespace": "wrong"}, expected=404)
        changed = call("PUT", f"/v1/contexts/{key}", {**context, "state": {"unused": False}})
        assert changed["snapshot"]["id"] != key
        assert call("GET", f"/v1/contexts?namespace=http-smoke&id_prefix={key[:8]}")[0]["id"] == key
        assert call("DELETE", f"/v1/contexts/{key}?namespace=http-smoke")["deleted"]
        assert not call("DELETE", f"/v1/contexts/{key}?namespace=http-smoke")["deleted"]
        call("POST", "/v1/decisions", request, expected=404)
        call("DELETE", "/v1/contexts?namespace=http-smoke")
        assert call("GET", "/v1/contexts?namespace=http-smoke") == []
    return {"result": "passed", "transport": "real HTTP/TCP", "health": health, "events": events}


def main():
    """Write a new evidence file and fail loudly rather than overwriting previous measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = exercise(args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(f"passed: {len(result['events'])} HTTP requests; 64-question fanout")


if __name__ == "__main__":
    main()
