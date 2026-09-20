"""A small command line for local decisions, resident serving and reproducible benchmarks."""

import argparse
import json
import sys
from pathlib import Path

from .backends import DEFAULT_MODEL, load_backend
from .runtime import Runtime
from .schemas import ContextSpec, DecisionRequest


def parser() -> argparse.ArgumentParser:
    """Keep backend configuration explicit and expose only implemented runtime modes."""
    result = argparse.ArgumentParser(prog="busbar")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("run", "serve", "benchmark", "evaluate", "profile"):
        command = sub.add_parser(name)
        command.add_argument("--backend", choices=("mlx", "hf", "vllm-metal"), default="mlx")
        command.add_argument("--model", default=DEFAULT_MODEL)
        command.add_argument("--revision", help="immutable SHA; known models have pinned defaults")
        command.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
        command.add_argument("--max-tokens", type=int, default=8192)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument(
            "--memory-fraction",
            type=float,
            default=0.35,
            help="vLLM Metal memory fraction; ignored by native backends",
        )
        if name == "run":
            command.add_argument("input", type=Path, help="JSON with context and questions")
            command.add_argument("--mode", choices=("cached", "fresh"), default="cached")
            command.add_argument(
                "--projection", choices=("auto", "selected", "full"), default="auto"
            )
        elif name == "serve":
            command.add_argument("--port", type=int, default=8787)
            command.add_argument("--cache-mib", type=int, default=1024)
            command.add_argument("--max-contexts", type=int, default=32)
            command.add_argument("--ttl", type=float, default=600)
        elif name in ("benchmark", "profile"):
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--context-tokens", type=int, default=4096)
            command.add_argument("--questions", type=int, default=16)
            command.add_argument("--repeats", type=int, default=3)
        else:
            command.add_argument("input", type=Path, help="SemIf-compatible labeled JSONL")
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--mode", choices=("cached", "fresh"), default="cached")
            command.add_argument(
                "--projection", choices=("auto", "selected", "full"), default="auto"
            )
    return result


def main():
    """Emit machine-readable results; invalid configuration fails with a concise nonzero exit."""
    args = parser().parse_args()
    try:
        if hasattr(args, "output") and args.output.exists():
            raise ValueError("benchmark output already exists; choose a new path")
        options = {"batch_size": args.batch_size, "max_tokens": args.max_tokens}
        if args.backend == "vllm-metal":
            options["memory_fraction"] = args.memory_fraction
        if args.dtype:
            options["dtype"] = args.dtype
        backend = load_backend(args.backend, args.model, args.revision, **options)
        if args.command == "serve":
            import uvicorn

            from .server import create_app

            runtime = Runtime(
                backend,
                max_contexts=args.max_contexts,
                max_cache_bytes=args.cache_mib * 1024 * 1024,
                ttl_seconds=args.ttl,
            )
            uvicorn.run(create_app(runtime), host="127.0.0.1", port=args.port, workers=1)
        elif args.command == "run":
            payload = json.loads(args.input.read_text())
            if set(payload) != {"context", "questions"}:
                raise ValueError("input requires exactly context and questions")
            runtime = Runtime(backend)
            context = ContextSpec.model_validate(payload["context"])
            compiled = runtime.compile_context(context)
            request = DecisionRequest(
                snapshot_id=compiled.snapshot.id,
                namespace=context.namespace,
                questions=payload["questions"],
                mode=args.mode,
                projection=args.projection,
            )
            decision = runtime.decide(request)
            print(
                json.dumps(
                    {"context": compiled.model_dump(), "result": decision.model_dump()}, indent=2
                )
            )
        else:
            if args.command == "evaluate":
                from .evaluation import evaluate

                report = evaluate(backend, args.input, args.mode, args.projection)
            elif args.command == "profile":
                from .profiling import profile

                report = profile(backend, args.context_tokens, args.questions, args.repeats)
            else:
                from .benchmark import benchmark

                report = benchmark(backend, args.context_tokens, args.questions, args.repeats)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as file:
                json.dump(report, file, indent=2)
                file.write("\n")
            print(json.dumps(report["summary"], indent=2))
    except (ValueError, OSError, ImportError) as exc:
        print(f"busbar: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
