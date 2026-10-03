"""Command-line entry point.

Three subcommands, deliberately thin. All behaviour lives in `pipeline` for
`migrate`, and in `ui.app` for `serve`; this module only parses arguments and
decides what to print.

    graft migrate path/to/file.py     convert, verify, print the report
    graft serve                       run the web UI on localhost
    graft doctor                      report what this machine can do

`migrate` writes the Markdown report to stdout by default, or to a file with
`-o`, so it composes with the rest of a shell instead of insisting on being
interactive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graft import __version__, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graft",
        description="CUDA to ROCm migration with correctness verification.",
    )
    parser.add_argument("--version", action="version", version=f"graft {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    migrate = sub.add_parser("migrate", help="convert and verify one file")
    migrate.add_argument("path", type=Path, help="a .py or .cu file")
    migrate.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the Markdown report here instead of stdout",
    )
    migrate.add_argument(
        "--json",
        action="store_true",
        help="print the summary as JSON instead of Markdown",
    )
    migrate.add_argument(
        "--no-parity",
        action="store_true",
        help="skip the parity check",
    )
    migrate.add_argument(
        "--no-bench",
        action="store_true",
        help="skip benchmarking",
    )
    migrate.add_argument(
        "--no-narrative",
        action="store_true",
        help="skip the optional agent layer",
    )

    serve = sub.add_parser("serve", help="run the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("doctor", help="report what this machine can do")
    return parser


def cmd_migrate(args: argparse.Namespace) -> int:
    from graft import pipeline

    if not args.path.is_file():
        print(f"error: no such file: {args.path}", file=sys.stderr)
        return 2

    result = pipeline.run(
        args.path,
        run_parity=not args.no_parity,
        run_bench=not args.no_bench,
        with_narrative=not args.no_narrative,
    )

    if args.json:
        from graft.ui.summary import summarize

        payload = summarize(result)
        payload.pop("report_text", None)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.report_text, encoding="utf-8")
        print(f"wrote {args.output}")
        for note in result.skipped:
            print(f"  skipped: {note}", file=sys.stderr)
        return 0

    print(result.report_text)
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    """Report capability, including what is missing.

    Exists so that "it didn't work" has a first question with a definite
    answer, rather than being discovered halfway through a migration.
    """
    import torch

    from graft.bench import telemetry
    from graft.core import runner

    snapshot = telemetry.device_snapshot()
    gpu = bool(torch.cuda.is_available())

    print(f"graft {__version__}")
    print(f"python        {sys.version.split()[0]}")
    print(f"torch         {torch.__version__}")
    print(f"device        {config.device_label()}")
    print(f"gpu available {'yes' if gpu else 'no'}")
    print(f"backend       {snapshot.get('backend', 'unknown')}")

    if gpu:
        vram = snapshot.get("total_vram_mb")
        print(f"total vram    {vram} MB" if vram else "total vram    not reported")
        print("\nBenchmarks can run. The verdict bands are still calibrated from")
        print("CPU noise and should be re-measured on this hardware.")
    else:
        print("\nNo AMD GPU detected. Everything works on CPU, but:")
        print("  - benchmarks are skipped, so no timing claim is possible")
        print("  - nothing here demonstrates ROCm behaviour")

    print("\nAgent layer")
    if config.LLM_MODEL and config.LLM_BASE_URL:
        print(f"  enabled: {config.LLM_MODEL} at {config.LLM_BASE_URL}")
    else:
        print("  disabled: set LLM_MODEL and LLM_BASE_URL to enable it")
    print("  the deterministic pipeline is unaffected either way")

    print("\nIsolation")
    for line in runner.sandbox_notes():
        print(f"  - {line}")

    if gpu:
        print("\nTo verify real HIP compilation, run the .cu example under hipcc:")
        print(f"  {config.EXAMPLES_DIR}/02_cuda_add_kernel.cu")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from graft.ui.app import serve

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"warning: binding {args.host}. The process that runs uploaded code "
            "shares your user account, filesystem, and network.",
            file=sys.stderr,
        )
    print(f"Graft on http://{args.host}:{args.port}")
    serve(host=args.host, port=args.port)
    return 0


COMMANDS = {
    "migrate": cmd_migrate,
    "serve": cmd_serve,
    "doctor": cmd_doctor,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
