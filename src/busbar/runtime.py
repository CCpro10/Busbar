"""Namespace-scoped immutable contexts, bounded KV retention and observable fanout."""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from .backends.base import Backend, PrefixState
from .compiler import Compiler, context_id, read_decision
from .schemas import CompileResult, ContextSpec, DecisionRequest, DecisionResult, Snapshot


class SnapshotNotFound(KeyError):
    """A snapshot was removed, expired, evicted or belongs to a different namespace."""


class CapacityError(ValueError):
    """One context cannot fit the configured retained KV budget."""


@dataclass
class _Entry:
    """Only cache policy timestamps mutate; prefix tokens and model state remain unchanged."""

    id: str
    namespace: str
    tokens: tuple[int, ...]
    prefix: PrefixState
    created_at: float
    last_used: float


class Runtime:
    """Compile once, retain across requests and branch into independently scored decisions."""

    def __init__(
        self,
        backend: Backend,
        *,
        max_contexts=32,
        max_cache_bytes=1_073_741_824,
        ttl_seconds=600.0,
    ):
        """Bound retained snapshots and serialize model operations for local accelerator safety."""
        if max_contexts < 1 or max_cache_bytes < 1 or not 0 < ttl_seconds < float("inf"):
            raise ValueError("cache count, byte budget and finite TTL must be positive")
        self.backend = backend
        self.compiler = getattr(backend, "compiler", None) or Compiler(backend.tokenizer)
        self.max_contexts = max_contexts
        self.max_cache_bytes = max_cache_bytes
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        # ponytail: one model operation at a time; a server scheduler belongs in a later backend.
        self._lock = threading.RLock()
        self._bytes = 0
        self._hits = self._misses = self._evictions = self._expirations = 0

    def _remove(self, key):
        """Keep byte accounting synchronized with all deletion and eviction paths."""
        self._bytes -= self._entries.pop(key).prefix.nbytes

    def _expire(self):
        """Use a monotonic sliding TTL so wall-clock changes do not corrupt lifetime policy."""
        deadline = time.monotonic() - self.ttl_seconds
        for key in [key for key, entry in self._entries.items() if entry.last_used <= deadline]:
            self._remove(key)
            self._expirations += 1

    def _entry(self, key, namespace):
        """Treat cross-namespace lookup exactly like an unknown ID."""
        self._expire()
        entry = self._entries.get(key)
        if entry is None or entry.namespace != namespace:
            raise SnapshotNotFound(key)
        return entry

    def _touch(self, entry):
        """Successful reuse refreshes both LRU priority and TTL."""
        entry.last_used = time.monotonic()
        self._entries.move_to_end(entry.id)

    def _snapshot(self, entry) -> Snapshot:
        """Expose metadata without leaking context text, input tokens or physical KV arrays."""
        return Snapshot(
            id=entry.id,
            namespace=entry.namespace,
            token_count=len(entry.tokens),
            cache_bytes=entry.prefix.nbytes,
            model=self.backend.identity["model"],
            revision=self.backend.identity["revision"],
            created_at=entry.created_at,
            expires_at=time.time() + max(0, entry.last_used + self.ttl_seconds - time.monotonic()),
        )

    def compile_context(self, spec: ContextSpec | dict) -> CompileResult:
        """Content-addressed creation is idempotent and reuses both tokens and materialized KV."""
        started = time.perf_counter()
        if not isinstance(spec, ContextSpec):
            spec = ContextSpec.model_validate(spec)
        with self._lock:
            self._expire()
            key = context_id(spec, self.backend.identity)
            entry = self._entries.get(key)
            reused = entry is not None
            if entry is not None:
                self._hits += 1
                self._touch(entry)
            else:
                tokens = self.compiler.context(spec)
                if len(tokens) >= self.backend.max_tokens:
                    raise ValueError("context exceeds token limit or leaves no room for a question")
                prefix = self.backend.prefill(tokens)
                if prefix.nbytes > self.max_cache_bytes:
                    raise CapacityError(
                        "context KV exceeds cache byte budget; no snapshot was stored"
                    )
                while self._entries and (
                    len(self._entries) >= self.max_contexts
                    or self._bytes + prefix.nbytes > self.max_cache_bytes
                ):
                    self._remove(next(iter(self._entries)))
                    self._evictions += 1
                entry = _Entry(key, spec.namespace, tokens, prefix, time.time(), time.monotonic())
                self._entries[key] = entry
                self._bytes += prefix.nbytes
                self._misses += 1
            return CompileResult(
                snapshot=self._snapshot(entry),
                reused=reused,
                prefill_tokens=0 if reused else len(entry.tokens),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

    def replace_context(self, old_id: str, spec: ContextSpec) -> CompileResult:
        """Create a new immutable version; the old version stays usable until normal eviction."""
        with self._lock:
            self._entry(old_id, spec.namespace)
            return self.compile_context(spec)

    def get_context(self, snapshot_id: str, namespace="default") -> Snapshot:
        """Inspect a live context without extending its TTL."""
        with self._lock:
            return self._snapshot(self._entry(snapshot_id, namespace))

    def list_contexts(self, namespace="default", id_prefix="") -> list[Snapshot]:
        """List live snapshots in one namespace, optionally filtered by ID prefix."""
        with self._lock:
            self._expire()
            return [
                self._snapshot(e)
                for e in self._entries.values()
                if e.namespace == namespace and e.id.startswith(id_prefix)
            ]

    def delete_context(self, snapshot_id: str, namespace="default") -> bool:
        """Remove a context idempotently; later decisions must recompile."""
        with self._lock:
            self._expire()
            entry = self._entries.get(snapshot_id)
            if entry is None or entry.namespace != namespace:
                return False
            self._remove(snapshot_id)
            return True

    def clear(self, namespace: str | None = None) -> int:
        """Release one namespace, or all runtime-held contexts and suffix tokens."""
        with self._lock:
            keys = [
                k for k, e in self._entries.items() if namespace is None or e.namespace == namespace
            ]
            for key in keys:
                self._remove(key)
            if namespace is None:
                self.compiler.clear()
                if hasattr(self.backend, "reset_cache"):
                    self.backend.reset_cache()
            return len(keys)

    def stats(self) -> dict:
        """Report live retained-cache usage and cumulative cache lifecycle counters."""
        with self._lock:
            self._expire()
            return {
                "contexts": len(self._entries),
                "cache_bytes": self._bytes,
                "max_cache_bytes": self.max_cache_bytes,
                "max_contexts": self.max_contexts,
                "ttl_seconds": self.ttl_seconds,
                "compile_hits": self._hits,
                "compile_misses": self._misses,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "model": self.backend.identity,
                "persistence": "process-memory",
                "scheduler": (
                    "serialized submissions; vLLM engine schedules fanout"
                    if self.backend.identity.get("backend") == "vllm-metal"
                    else "serialized; length-grouped fanout"
                ),
            }

    def decide(self, request: DecisionRequest | dict) -> DecisionResult:
        """Validate the complete fanout before execution; return no partial answers."""
        started = time.perf_counter()
        if not isinstance(request, DecisionRequest):
            request = DecisionRequest.model_validate(request)
        with self._lock:
            entry = self._entry(request.snapshot_id, request.namespace)
            mark = time.perf_counter()
            compiled = [self.compiler.question(q) for q in request.questions.values()]
            paths = [path for question in compiled for path in question.paths]
            if any(
                len(entry.tokens) + len(path.token_ids) > self.backend.max_tokens for path in paths
            ):
                raise ValueError(
                    "context plus question exceeds token limit; input was not truncated"
                )
            cached = request.mode == "cached"
            projection = request.projection
            if projection == "auto":
                projection = getattr(self.backend, "default_projection", "selected")
            sequences = [p.token_ids if cached else entry.tokens + p.token_ids for p in paths]
            compile_ms = (time.perf_counter() - mark) * 1000
            mark = time.perf_counter()
            output = self.backend.score(
                sequences,
                [p.label_ids for p in paths],
                entry.prefix if cached else None,
                projection,
            )
            inference_ms = (time.perf_counter() - mark) * 1000
            if len(output.logits) != len(paths) or any(
                len(scores) != len(path.label_ids)
                for scores, path in zip(output.logits, paths, strict=True)
            ):
                raise RuntimeError("backend returned the wrong number of readout scores")
            # Group scalar candidate paths back into one categorical decision.
            grouped, offset = [], 0
            for question in compiled:
                end = offset + len(question.paths)
                grouped.append([value for scores in output.logits[offset:end] for value in scores])
                offset = end
            decisions = {
                name: read_decision(question, logits).model_copy(
                    update={"logit_space": output.details.get("logit_space", "raw_logits")}
                )
                for (name, question), logits in zip(request.questions.items(), grouped, strict=True)
            }
            self._touch(entry)
            return DecisionResult(
                snapshot_id=entry.id,
                decisions=decisions,
                mode=request.mode,
                projection=projection,
                prefix_tokens=len(entry.tokens),
                reused_prefix_tokens=(
                    output.reused_tokens
                    if output.reused_tokens is not None
                    else len(entry.tokens) * len(paths)
                    if cached
                    else 0
                ),
                computed_tokens=output.computed_tokens,
                batches=output.batches,
                compile_ms=compile_ms,
                inference_ms=inference_ms,
                total_ms=(time.perf_counter() - started) * 1000,
                backend_details=output.details,
                probability_status=output.details.get(
                    "probability_status", "conditional label probabilities; uncalibrated"
                ),
            )
