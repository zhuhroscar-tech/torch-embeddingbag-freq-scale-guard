"""Command-line interface: run the from-scratch diagnosis of the MPS
embedding_bag(scale_grad_by_freq=True) silent gradient bug against the
currently installed torch build, using the shared semantic-color
design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-embeddingbag-freq-scale-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch.nn.functional.embedding_bag(..., scale_grad_by_freq=True) "
            "silently ignores the flag on the MPS (Apple Silicon GPU) "
            "backend (pytorch/pytorch#190061), and verify the "
            "safe_embedding_bag() guard applies correct 1/frequency "
            "gradient scaling regardless of backend. Runs on CPU always; "
            "runs on MPS only when this host's MPS device passes a real "
            "allocation probe (not just torch.backends.mps.is_available()). "
            "Never trusts a cached or previously-reported result, always "
            "re-runs the actual repro on THIS host's installed torch build."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-embeddingbag-freq-scale-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields(
        [
            ("torch version", report["torch_version"]),
            ("mps functional on this host", "yes" if report["mps_functional"] else "no (see notes below)"),
        ]
    )

    if report["any_native_silently_wrong"]:
        print(status_headline(style, "fail", "MPS embedding_bag(scale_grad_by_freq=True) silent bug reproduced on this host"))
    else:
        print(status_headline(style, "info", "no silent scale_grad_by_freq bug reproduced on this host's exercised devices"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "guard matches the CPU-oracle 1/frequency scaling contract on every exercised device"))
    else:
        print(status_headline(style, "fail", "guard did NOT match the expected contract on at least one device"))

    section("per-device results (native vs guard vs CPU oracle, row for the doubly-occurring index)")
    for c in report["cases"]:
        if not c["ran"]:
            print_fields([(c["device"], f"skipped: {c['skip_reason']}")])
            continue
        native_flag = "SILENT-WRONG" if c["native_matches_oracle"] is False else "ok"
        guard_flag = "guard-ok" if c["guard_matches_oracle"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["device"],
                    f"native={native_flag:12s}  guard={guard_flag:12s}  "
                    f"native_grad={c['native_grad_row1']}  guard_grad={c['guard_grad_row1']}  "
                    f"oracle={c['cpu_oracle_grad_row1']}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
