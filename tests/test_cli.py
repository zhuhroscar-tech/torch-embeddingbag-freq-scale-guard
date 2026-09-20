"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is
installed. Mocks core.diagnose so every CLI message path (torch-
unavailable, no-bug "info" line, guard-mismatch "fail" line) is
exercised deterministically, regardless of whether this host's
installed torch build / MPS device happens to reproduce the
underlying bug."""
from __future__ import annotations

import json
import runpy
import sys

import pytest

from torch_embeddingbag_freq_scale_guard import core
from torch_embeddingbag_freq_scale_guard.cli import main


def _fake_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/190061",
        "mps_functional": True,
        "cases": [
            {
                "device": "cpu",
                "ran": True,
                "skip_reason": None,
                "native_grad_row1": [1.0, 1.0, 1.0],
                "guard_grad_row1": [1.0, 1.0, 1.0],
                "cpu_oracle_grad_row1": [1.0, 1.0, 1.0],
                "native_matches_oracle": True,
                "guard_matches_oracle": True,
            },
            {
                "device": "mps",
                "ran": True,
                "skip_reason": None,
                "native_grad_row1": [2.0, 2.0, 2.0],
                "guard_grad_row1": [1.0, 1.0, 1.0],
                "cpu_oracle_grad_row1": [1.0, 1.0, 1.0],
                "native_matches_oracle": False,
                "guard_matches_oracle": True,
            },
        ],
        "any_native_silently_wrong": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-embeddingbag-freq-scale-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert "guard_fully_correct" in report
    assert code in (0, 1)


def test_json_exit_code_matches_guard_fully_correct(capsys):
    pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert code == (0 if report["guard_fully_correct"] else 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_text_output_reports_per_device_section(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "per-device results" in out


def test_torch_unavailable_json_mode_reports_error_and_exit_2(monkeypatch, capsys):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis"}
    assert code == 2


def test_torch_unavailable_text_mode_reports_fail_headline_and_exit_2(monkeypatch, capsys):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch unavailable: torch is required for diagnosis" in out
    assert "[X]" in out
    assert code == 2


def test_no_bug_prints_info_line(monkeypatch, capsys):
    monkeypatch.setattr(
        core,
        "diagnose",
        lambda: _fake_report(
            any_native_silently_wrong=False,
            cases=[
                {
                    "device": "cpu",
                    "ran": True,
                    "skip_reason": None,
                    "native_grad_row1": [1.0, 1.0, 1.0],
                    "guard_grad_row1": [1.0, 1.0, 1.0],
                    "cpu_oracle_grad_row1": [1.0, 1.0, 1.0],
                    "native_matches_oracle": True,
                    "guard_matches_oracle": True,
                }
            ],
        ),
    )
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "no silent scale_grad_by_freq bug reproduced" in out


def test_bug_reproduced_prints_fail_line(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(any_native_silently_wrong=True))
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "MPS embedding_bag(scale_grad_by_freq=True) silent bug reproduced on this host" in out


def test_guard_mismatch_prints_fail_line_and_exit_1(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "guard did NOT match the expected contract" in out
    assert "guard matches the CPU-oracle" not in out
    assert code == 1

    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    code_json = main(["--json"])
    capsys.readouterr()
    assert code_json == 1


def test_skipped_device_prints_skip_reason(monkeypatch, capsys):
    monkeypatch.setattr(
        core,
        "diagnose",
        lambda: _fake_report(
            mps_functional=False,
            cases=[
                {
                    "device": "cpu",
                    "ran": True,
                    "skip_reason": None,
                    "native_grad_row1": [1.0, 1.0, 1.0],
                    "guard_grad_row1": [1.0, 1.0, 1.0],
                    "cpu_oracle_grad_row1": [1.0, 1.0, 1.0],
                    "native_matches_oracle": True,
                    "guard_matches_oracle": True,
                },
                {
                    "device": "mps",
                    "ran": False,
                    "skip_reason": "mps not available on this host",
                    "native_grad_row1": None,
                    "guard_grad_row1": None,
                    "cpu_oracle_grad_row1": [1.0, 1.0, 1.0],
                    "native_matches_oracle": None,
                    "guard_matches_oracle": None,
                },
            ],
        ),
    )
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "skipped: mps not available on this host" in out


def test_module_entry_point_runs_main_and_exits_with_its_code(monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    monkeypatch.setattr(sys, "argv", ["torch-embeddingbag-freq-scale-guard", "--no-color"])
    monkeypatch.delitem(sys.modules, "torch_embeddingbag_freq_scale_guard.cli", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("torch_embeddingbag_freq_scale_guard.cli", run_name="__main__")
    assert exc_info.value.code == 1
