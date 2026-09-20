"""Local HTTP API around one resident runtime; no model loading occurs at module import."""

import logging

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from . import __version__
from .backends.base import BackendOutputError
from .runtime import CapacityError, Runtime, SnapshotNotFound
from .schemas import CompileResult, ContextSpec, DecisionRequest, DecisionResult, Snapshot


def create_app(runtime: Runtime) -> FastAPI:
    """Expose the full snapshot lifecycle and document a stable versioned decision contract."""
    app = FastAPI(
        title="Busbar", version=__version__, description="Local context-reuse decision runtime"
    )

    _register_errors(app)
    app.include_router(_status_routes(runtime))
    app.include_router(_context_routes(runtime))
    app.include_router(_decision_routes(runtime))
    return app


def _register_errors(app: FastAPI):
    """Keep input, lifecycle and backend failures distinct for callers and server logs."""

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

    @app.exception_handler(BackendOutputError)
    async def invalid_backend_output(request: Request, exc: BackendOutputError):
        """Report scorer failures separately and keep diagnostics in server logs."""
        logging.getLogger(__name__).error("Decision backend contract failed", exc_info=exc)
        return JSONResponse(
            status_code=502, content={"detail": "backend returned invalid decision output"}
        )


def _status_routes(runtime: Runtime) -> APIRouter:
    """Expose model readiness and cache observations without modifying snapshots."""
    router = APIRouter()

    @router.get("/health")
    def health():
        """The app exists only after a backend has loaded successfully."""
        return {"status": "ready", "backend": runtime.backend.identity}

    @router.get("/v1/stats")
    def stats():
        """Expose cache accounting without returning context text."""
        return runtime.stats()

    return router


def _context_routes(runtime: Runtime) -> APIRouter:
    """Own only the namespace-scoped context lifecycle endpoints."""
    router = APIRouter()

    @router.post("/v1/contexts", response_model=CompileResult)
    def compile_context(spec: ContextSpec):
        """Create or reuse a content-addressed context snapshot."""
        return runtime.compile_context(spec)

    @router.get("/v1/contexts", response_model=list[Snapshot])
    def list_contexts(
        namespace: str = "default", id_prefix: str = Query(default="", max_length=64)
    ):
        """Filter live context metadata within one namespace."""
        return runtime.list_contexts(namespace, id_prefix)

    @router.get("/v1/contexts/{snapshot_id}", response_model=Snapshot)
    def get_context(snapshot_id: str, namespace: str = "default"):
        """Read metadata without keeping an idle cache alive."""
        return runtime.get_context(snapshot_id, namespace)

    @router.put("/v1/contexts/{snapshot_id}", response_model=CompileResult)
    def replace_context(snapshot_id: str, spec: ContextSpec):
        """Version a context rather than mutating KV already referenced by another caller."""
        return runtime.replace_context(snapshot_id, spec)

    @router.delete("/v1/contexts/{snapshot_id}")
    def delete_context(snapshot_id: str, namespace: str = "default"):
        """Idempotently remove the selected context."""
        return {"deleted": runtime.delete_context(snapshot_id, namespace)}

    @router.delete("/v1/contexts")
    def clear_contexts(namespace: str = "default"):
        """Release every retained context in one namespace."""
        return {"deleted_count": runtime.clear(namespace)}

    return router


def _decision_routes(runtime: Runtime) -> APIRouter:
    """Run validated decision requests against the injected resident runtime."""
    router = APIRouter()

    @router.post("/v1/decisions", response_model=DecisionResult)
    def decide(request: DecisionRequest):
        """Run all validated questions against one immutable snapshot."""
        return runtime.decide(request)

    return router
