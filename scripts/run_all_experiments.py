"""
Run all Step 9 experiment scripts sequentially.

Order:
  1. compare_generators.py      — 20q SQuAD, generator comparison
  2. evaluate_mistral_grader.py — 100q SQuAD, grader ablation
  3. prepare_hotpot.py          — build HotpotQA index (one-time)
  4. evaluate_hotpot.py         — 200q HotpotQA, cross-dataset eval

Usage:
  python scripts/run_all_experiments.py                  # full run
  python scripts/run_all_experiments.py --smoke-test     # 2q sanity check first
  python scripts/run_all_experiments.py --start 3        # resume from prepare_hotpot
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

# Fix Windows charmap errors when scripts print Unicode text (e.g. HotpotQA titles).
# PYTHONIOENCODING only sets Python's own stdout/stderr encoding — it does NOT affect
# how subprocess.PIPE reads child process output, avoiding the _readerthread crash
# that PYTHONUTF8=1 causes (which forces UTF-8 on all pipe reads, including cp1252 output
# from system tools like git or bitsandbytes on Windows).
_CHILD_ENV = os.environ.copy()
_CHILD_ENV["PYTHONIOENCODING"] = "utf-8"

STEPS = [
    {
        "name": "compare_generators",
        "script": "scripts/compare_generators.py",
        "desc": "Phase 3 — Generator comparison (20q SQuAD)",
        "queries_flag": "--num-queries",
        "full_n": 20,
    },
    {
        "name": "evaluate_mistral_grader",
        "script": "scripts/evaluate_mistral_grader.py",
        "desc": "Phase 4 — Grader ablation (100q SQuAD)",
        "queries_flag": "--num-queries",
        "full_n": 100,
    },
    {
        "name": "prepare_hotpot",
        "script": "scripts/prepare_hotpot.py",
        "desc": "Phase 5 — Build HotpotQA index",
        "queries_flag": "--num-questions",
        "full_n": 200,
        "no_plots": True,  # script has no --skip-plots flag
    },
    {
        "name": "evaluate_hotpot",
        "script": "scripts/evaluate_hotpot.py",
        "desc": "Phase 6 — HotpotQA benchmark (200q)",
        "queries_flag": "--num-questions",
        "full_n": 200,
    },
]

SMOKE_N = 2


def fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def run_step(step: dict, step_num: int, total: int, smoke: bool = False) -> bool:
    """Run one script as a subprocess. Returns True on success."""
    script = step["script"]
    desc = step["desc"]

    cmd = [sys.executable, script]
    if smoke:
        cmd += [step["queries_flag"], str(SMOKE_N)]
        if not step.get("no_plots"):
            cmd.append("--skip-plots")
        desc += f"  [SMOKE: {SMOKE_N}q]"

    print(f"\n{'='*60}")
    print(f"[{step_num}/{total}] {desc}")
    print(f"      {' '.join(cmd)}")
    print(f"      started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, env=_CHILD_ENV)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n[OK] {step['name']} completed in {fmt_duration(elapsed)}\n")
        return True
    else:
        print(
            f"\n[FAILED] {step['name']} exited with code {result.returncode} "
            f"after {fmt_duration(elapsed)}\n"
        )
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all Step 9 experiment scripts")
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        metavar="N",
        help="Start from step N (1=compare_generators, 2=evaluate_mistral_grader, "
        "3=prepare_hotpot, 4=evaluate_hotpot). Default: 1",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Run a {SMOKE_N}-query sanity check on every step before the full run.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        default=True,
        help="Abort remaining steps if one fails (default: True)",
    )
    args = parser.parse_args()

    steps_to_run = STEPS[args.start - 1 :]
    total = len(steps_to_run)

    print(f"\nStep 9 — Running {total} experiment script(s) sequentially")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps:   {', '.join(s['name'] for s in steps_to_run)}")
    if args.smoke_test:
        print(f"Mode:    SMOKE TEST ({SMOKE_N}q each) → then full run")

    overall_start = time.time()
    results = {}

    # --- Smoke test pass ---
    if args.smoke_test:
        print(f"\n{'#'*60}")
        print(f"SMOKE TEST — {SMOKE_N} queries per script")
        print(f"{'#'*60}")
        for i, step in enumerate(steps_to_run, start=args.start):
            ok = run_step(step, i, len(STEPS), smoke=True)
            if not ok:
                print(
                    f"\nSmoke test FAILED on {step['name']}. "
                    f"Fix the issue before running the full experiment.\n"
                )
                sys.exit(1)
        print(f"\n{'#'*60}")
        print("Smoke test passed. Starting full run...")
        print(f"{'#'*60}\n")

    # --- Full run ---
    for i, step in enumerate(steps_to_run, start=args.start):
        ok = run_step(step, i, len(STEPS), smoke=False)
        results[step["name"]] = "OK" if ok else "FAILED"

        if not ok and args.stop_on_failure:
            print(f"Aborting: {step['name']} failed. Re-run with --start {i} to retry.\n")
            break

    # Summary
    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"SUMMARY  (total: {fmt_duration(total_elapsed)})")
    print(f"{'='*60}")
    for name, status in results.items():
        marker = "✓" if status == "OK" else "✗"
        print(f"  {marker}  {name}: {status}")

    skipped = [s["name"] for s in STEPS if s["name"] not in results]
    for name in skipped:
        print(f"  -  {name}: skipped")

    print()
    all_ok = all(v == "OK" for v in results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
