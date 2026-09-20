"""A small command line for local decisions, resident serving and reproducible benchmarks."""

import argparse
import json
import sys
from pathlib import Path

from .backends import DEFAULT_MODEL, load_backend
from .runtime import Runtime
from .schemas import DecisionInput, DecisionRequest
from .storage import write_json


def parser() -> argparse.ArgumentParser:
    """Keep backend configuration explicit and expose only implemented runtime modes."""
    result = argparse.ArgumentParser(prog="busbar")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("run", "serve", "benchmark", "evaluate", "profile"):
        command = sub.add_parser(name)
        command.add_argument("--backend", choices=("mlx", "hf", "vllm-metal"), default="mlx")
        command.add_argument(
            "--model", help=f"default: {DEFAULT_MODEL}, or the head manifest model"
        )
        command.add_argument("--head", type=Path, help="trained head directory; native MLX only")
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
                "--projection", choices=("auto", "selected", "full", "head"), default="auto"
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
            command.add_argument("input", type=Path, help="labeled JSONL")
            command.add_argument("--format", choices=("semif", "native"), default="semif")
            command.add_argument(
                "--allow-training-data",
                action="store_true",
                help="explicitly measure fit on known train/validation examples",
            )
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--mode", choices=("cached", "fresh"), default="cached")
            command.add_argument(
                "--projection", choices=("auto", "selected", "full", "head"), default="auto"
            )
    train = sub.add_parser(
        "train-head", help="train a native MLX selection head over a frozen base"
    )
    train.add_argument("input", type=Path, help="native labeled training JSONL")
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True, help="new immutable version directory")
    train.add_argument("--model")
    train.add_argument("--revision")
    train.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    train.add_argument(
        "--init-head", type=Path, help="warm start weights; starts a fresh optimizer"
    )
    train.add_argument("--max-tokens", type=int, default=8192)
    train.add_argument("--batch-size", type=int, default=8, help="backbone candidate batch size")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--learning-rate", type=float, default=0.003)
    train.add_argument("--train-batch-size", type=int, default=32)
    train.add_argument("--feature-cache-mib", type=int, default=256)
    train.add_argument("--seed", type=int, default=42)
    return result


def _load_backend(args):
    """Translate shared CLI options at one boundary while keeping optional imports lazy."""
    options = {"batch_size": args.batch_size, "max_tokens": args.max_tokens}
    if args.backend == "vllm-metal":
        options["memory_fraction"] = args.memory_fraction
    if args.dtype:
        options["dtype"] = args.dtype
    return load_backend(args.backend, args.model, args.revision, head=args.head, **options)


def _run_command(args):
    """Validate all one-shot questions before loading a model or creating context state."""
    payload = DecisionInput.model_validate_json(args.input.read_bytes())
    runtime = Runtime(_load_backend(args))
    context = runtime.compile_context(payload.context)
    decision = runtime.decide(
        DecisionRequest(
            snapshot_id=context.snapshot.id,
            namespace=payload.context.namespace,
            questions=payload.questions,
            mode=args.mode,
            projection=args.projection,
        )
    )
    return {"context": context.model_dump(), "result": decision.model_dump()}


def _serve_command(args):
    """Serve one resident runtime; the worker count must preserve ownership of its cache."""
    import uvicorn

    from .server import create_app

    runtime = Runtime(
        _load_backend(args),
        max_contexts=args.max_contexts,
        max_cache_bytes=args.cache_mib * 1024 * 1024,
        ttl_seconds=args.ttl,
    )
    uvicorn.run(create_app(runtime), host="127.0.0.1", port=args.port, workers=1)


def _evaluate_command(args):
    """Preflight the whole dataset before model loading; reuse those exact rows for evaluation."""
    from .evaluation import evaluate, prepare_evaluation

    inputs = prepare_evaluation(args.input, args.format)
    if args.allow_training_data and args.format != "native":
        raise ValueError("training-data override is native-only")
    return evaluate(
        _load_backend(args),
        inputs,
        args.mode,
        args.projection,
        data_format=args.format,
        allow_training_data=args.allow_training_data,
    )


def _measure_command(args):
    """Choose an implemented latency experiment without duplicating backend setup."""
    from .benchmark import benchmark
    from .profiling import profile

    if args.command == "profile" and (args.head or args.backend != "mlx"):
        raise ValueError("vocabulary phase profiling requires the native MLX vocabulary scorer")
    measure = profile if args.command == "profile" else benchmark
    return measure(_load_backend(args), args.context_tokens, args.questions, args.repeats)


def _train_command(args):
    """Keep training configuration validation separate from the accelerator implementation."""
    from .training import TrainingConfig, train_head

    config = TrainingConfig(**{name: getattr(args, name) for name in TrainingConfig.model_fields})
    return train_head(
        args.input,
        args.validation,
        args.output,
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        init_head=args.init_head,
        config=config,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )


def main(argv=None):
    """Dispatch a command and publish complete reports; expected input errors exit with code 2."""
    args = parser().parse_args(argv)
    handlers = {
        "run": _run_command,
        "serve": _serve_command,
        "evaluate": _evaluate_command,
        "train-head": _train_command,
        "benchmark": _measure_command,
        "profile": _measure_command,
    }
    try:
        if hasattr(args, "output") and args.output.exists():
            raise ValueError("output already exists; choose a new path or version directory")
        report = handlers[args.command](args)
        if report is None:
            return
        if args.command in ("evaluate", "benchmark", "profile"):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, report)
            report = report["summary"]
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError) as exc:
        print(f"busbar: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
