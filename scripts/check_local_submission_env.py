#!/usr/bin/env python
"""Check the local toolchain needed for reproducible Kaggle submissions.

The check is intentionally read-only. It never prints the Kaggle token value.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


COMPETITION = "llm-agentic-legal-information-retrieval"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_command(args: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError:
        return 127, f"not found: {args[0]}"
    except OSError as exc:
        return 126, f"failed to execute {args[0]}: {exc}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {' '.join(args)}"
    return proc.returncode, proc.stdout.strip()


def version_tuple(text: str) -> tuple[int, ...]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return ()
    return tuple(int(part) for part in match.groups(default="0"))


def check_python(min_version: tuple[int, int]) -> CheckResult:
    current = sys.version_info[:3]
    detail = f"{sys.executable} ({sys.version.split()[0]})"
    return CheckResult("python", current >= min_version, detail)


def check_git(min_version: tuple[int, int]) -> CheckResult:
    git = shutil.which("git")
    if not git:
        return CheckResult("git", False, "git not found on PATH")
    code, out = run_command(["git", "--version"])
    ok = code == 0 and version_tuple(out) >= min_version
    return CheckResult("git", ok, f"{git}; {out}")


def check_proxy(expected_port: str) -> CheckResult:
    values = {
        key: os.environ.get(key, "")
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    }
    matched = [key for key, value in values.items() if expected_port in value]
    detail = ", ".join(f"{key}={'set' if value else 'missing'}" for key, value in values.items())
    return CheckResult("proxy", bool(matched), f"{detail}; expected port {expected_port}")


def check_token() -> CheckResult:
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        return CheckResult("kaggle_token", False, "KAGGLE_API_TOKEN missing")
    shape = "KGAT_*" if token.startswith("KGAT_") else "set"
    return CheckResult("kaggle_token", True, shape)


def check_kaggle_cli(check_remote: bool) -> CheckResult:
    kaggle = shutil.which("kaggle")
    if not kaggle:
        return CheckResult("kaggle_cli", False, "kaggle not found on PATH")
    code, out = run_command(["kaggle", "--version"])
    if code != 0:
        return CheckResult("kaggle_cli", False, f"{kaggle}; {out}")
    if not check_remote:
        return CheckResult("kaggle_cli", True, f"{kaggle}; {out}")
    code, remote = run_command(
        ["kaggle", "competitions", "list", "-s", COMPETITION],
        timeout=120,
    )
    ok = code == 0 and COMPETITION in remote
    status = "competition reachable" if ok else "competition lookup failed"
    return CheckResult("kaggle_cli", ok, f"{kaggle}; {out}; {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-remote", action="store_true", help="Call Kaggle to verify competition access.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--proxy-port", default="7897")
    parser.add_argument("--min-python", default="3.11")
    parser.add_argument("--min-git", default="2.54")
    args = parser.parse_args()

    min_python = tuple(int(part) for part in args.min_python.split(".")[:2])
    min_git = tuple(int(part) for part in args.min_git.split(".")[:2])

    results = [
        check_python(min_python),
        check_git(min_git),
        check_proxy(args.proxy_port),
        check_token(),
        check_kaggle_cli(args.check_remote),
    ]

    payload = {
        "ok": all(result.ok for result in results),
        "checks": [result.__dict__ for result in results],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for result in results:
            marker = "PASS" if result.ok else "FAIL"
            print(f"[{marker}] {result.name}: {result.detail}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
