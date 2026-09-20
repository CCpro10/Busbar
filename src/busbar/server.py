"""Local HTTP API around one resident runtime; no model loading occurs at module import."""

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from . import __version__
from .runtime import CapacityError, Runtime, SnapshotNotFound
from .schemas import CompileResult, ContextSpec, DecisionRequest, DecisionResult, Snapshot


def create_app(runtime: Runtime) -> FastAPI:
    """Expose the full snapshot lifecycle and document a stable versioned decision contract."""
    app = FastAPI(
        title="Busbar", version=__version__, description="Local context-reuse decision runtime"
    )

    @app.exception_handler(SnapshotNotFound)
    async def missing_snapshot(request: Request, exc: SnapshotNotFound):
        """Unknown, expired and cross-namespace IDs share the same response."""
        return JSONResponse(
            status_code=404, content={"detail": "snapshot unavailable; compile context again"}
        )

    @app.exception_handler(CapacityError)
    async def capacity_error(request: Request, exc: CapacityError):
        """Tell the caller that its one context cannot fit the configured budget."""
        return JSONResponse(status_code=413, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid_input(request: Request, exc: ValueError):
        """Model-bound token and label checks are input errors, not silent truncation."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        """The app exists only after a backend has loaded successfully."""
        return {"status": "ready", "backend": runtime.backend.identity}

    @app.get("/v1/stats")
    def stats():
        """Expose cache accounting without returning context text."""
        return runtime.stats()

    @app.post("/v1/contexts", response_model=CompileResult)
    def compile_context(spec: ContextSpec):
        """Create or reuse a content-addressed context snapshot."""
        return runtime.compile_context(spec)

    @app.get("/v1/contexts", response_model=list[Snapshot])
    def list_contexts(
        namespace: str = "default", id_prefix: str = Query(default="", max_length=64)
    ):
        """Filter live context metadata within one namespace."""
        return runtime.list_contexts(namespace, id_prefix)

    @app.get("/v1/contexts/{snapshot_id}", response_model=Snapshot)
    def get_context(snapshot_id: str, namespace: str = "default"):
        """Read metadata without keeping an idle cache alive."""
        return runtime.get_context(snapshot_id, namespace)

    @app.put("/v1/contexts/{snapshot_id}", response_model=CompileResult)
    def replace_context(snapshot_id: str, spec: ContextSpec):
        """Version a context rather than mutating KV already referenced by another caller."""
        return runtime.replace_context(snapshot_id, spec)

    @app.delete("/v1/contexts/{snapshot_id}")
    def delete_context(snapshot_id: str, namespace: str = "default"):
        """Idempotently remove the selected context."""
        return {"deleted": runtime.delete_context(snapshot_id, namespace)}

    @app.delete("/v1/contexts")
    def clear_contexts(namespace: str = "default"):
        """Release every retained context in one namespace."""
        return {"deleted_count": runtime.clear(namespace)}

    @app.post("/v1/decisions", response_model=DecisionResult)
    def decide(request: DecisionRequest):
        """Run all validated questions against one immutable snapshot."""
        return runtime.decide(request)

    return app
