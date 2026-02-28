#!/usr/bin/env python3
"""Long-session test runner for PCDC characterisation (100-200+ turns).

Sends prompts from data/long_session_prompts.json, collects pcdc telemetry,
and writes structured JSON output for post-hoc analysis.

Usage:
    python scripts/long_session_test.py --scenario full
    python scripts/long_session_test.py --scenario repetition_probe --delay 1.0
    python scripts/long_session_test.py --scenario all --fail-fast
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROMPTS_FILE = PROJECT_DIR / "data" / "long_session_prompts.json"

SCENARIO_ORDER = [
    "sustained_domain",
    "rapid_switching",
    "gradual_drift",
    "return_to_origin",
    "complexity_ladder",
    "repetition_probe",
]


def load_prompts(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_prompt_list(corpus: dict, scenario: str) -> list[dict]:
    """Build ordered prompt list based on selected scenario mode."""
    if scenario == "full":
        prompts = build_prompt_list(corpus, "all")
        prompts += build_prompt_list(corpus, "general")
        return prompts
    if scenario == "all":
        prompts = []
        for name in SCENARIO_ORDER:
            for p in corpus["scenarios"][name]["prompts"]:
                prompts.append({**p, "scenario": name})
        return prompts
    if scenario == "general":
        return [{**p, "scenario": "general"} for p in corpus["general"]]
    if scenario in corpus["scenarios"]:
        return [
            {**p, "scenario": scenario}
            for p in corpus["scenarios"][scenario]["prompts"]
        ]
    print(f"Unknown scenario: {scenario}", file=sys.stderr)
    print(f"Available: {', '.join(SCENARIO_ORDER)} | all | general | full")
    sys.exit(1)


def send_prompt(
    url: str, text: str, max_tokens: int, timeout: int = 180
) -> dict | None:
    """Send a single prompt to the PCDC server and return the response dict."""
    body = json.dumps({
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_stats(url: str) -> dict | None:
    """Fetch /v1/pcdc/stats from the server."""
    try:
        with urllib.request.urlopen(f"{url}/v1/pcdc/stats", timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  Warning: could not fetch stats: {e}", file=sys.stderr)
        return None


def format_progress(
    turn: int, total: int, scenario: str, domain: str, pcdc: dict | None, elapsed: float
) -> str:
    """Format a single progress line."""
    if pcdc:
        er = pcdc.get("reconstruction_energy", 0)
        ep = pcdc.get("predictive_energy", 0)
        cos = pcdc.get("cosine_distance")
        dev = pcdc.get("deviation_match_score")
        temp = pcdc.get("adjusted_temperature", 0)
        cos_s = f"{cos:.3f}" if cos is not None else "N/A"
        dev_s = f"{dev:.3f}" if dev is not None else "N/A"
        return (
            f"[{turn:>3}/{total}] {scenario}/{domain:<14} "
            f"E_r={er:.1f} E_p={ep:.1f} cos={cos_s} "
            f"dev={dev_s} T={temp:.3f} ({elapsed:.1f}s)"
        )
    return f"[{turn:>3}/{total}] {scenario}/{domain:<14} ERROR ({elapsed:.1f}s)"


def run_session(
    url: str,
    prompts: list[dict],
    max_tokens: int,
    delay: float,
    fail_fast: bool,
    label: str,
) -> dict:
    """Run the full session, returning the results dict."""
    total = len(prompts)
    started = datetime.now().isoformat()
    turns = []
    errors = 0

    print(f"Starting session: {label}")
    print(f"  URL: {url}")
    print(f"  Turns: {total}, delay: {delay}s, max_tokens: {max_tokens}")
    print()

    for i, entry in enumerate(prompts, 1):
        text = entry["text"]
        scenario = entry["scenario"]
        domain = entry["domain"]
        complexity = entry["complexity"]

        t0 = time.time()
        pcdc_data = None
        reply_preview = None

        reply_full = None
        try:
            resp = send_prompt(url, text, max_tokens)
            pcdc_data = resp.get("pcdc", {})
            reply_full = resp["choices"][0]["message"]["content"]
            reply_preview = reply_full[:120]
        except (urllib.error.URLError, Exception) as e:
            errors += 1
            print(f"  ERROR on turn {i}: {e}", file=sys.stderr)
            if fail_fast:
                print("  --fail-fast set, aborting.", file=sys.stderr)
                break

        elapsed = time.time() - t0

        turn_record = {
            "turn": i,
            "scenario": scenario,
            "domain": domain,
            "complexity": complexity,
            "prompt": text,
            "timestamp": t0,
            "elapsed_s": round(elapsed, 3),
            "pcdc": pcdc_data,
            "reply": reply_full,
            "reply_preview": reply_preview,
        }
        turns.append(turn_record)

        print(format_progress(i, total, scenario, domain, pcdc_data, elapsed))

        if i < total:
            time.sleep(delay)

    # Fetch final stats
    print("\nFetching server stats...")
    server_stats = fetch_stats(url)

    finished = datetime.now().isoformat()
    result = {
        "meta": {
            "label": label,
            "started": started,
            "finished": finished,
            "total_turns": len(turns),
            "errors": errors,
            "url": url,
            "max_tokens": max_tokens,
            "delay": delay,
        },
        "turns": turns,
        "server_stats": server_stats,
    }

    print(f"\nSession complete: {len(turns)} turns, {errors} errors")
    return result


def main():
    parser = argparse.ArgumentParser(description="PCDC long-session test runner")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument(
        "--scenario", default="full",
        help="Scenario to run: all | general | full | <scenario_name>",
    )
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests (seconds)")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max tokens per response")
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: results/session_YYYYMMDD_HHMMSS.json)",
    )
    parser.add_argument("--label", default="long-session", help="Session label for metadata")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on first error")
    args = parser.parse_args()

    corpus = load_prompts(PROMPTS_FILE)
    prompts = build_prompt_list(corpus, args.scenario)

    if not prompts:
        print("No prompts selected.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(prompts)} prompts for scenario '{args.scenario}'")

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_DIR / "results" / f"session_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_session(
        url=args.url,
        prompts=prompts,
        max_tokens=args.max_tokens,
        delay=args.delay,
        fail_fast=args.fail_fast,
        label=args.label,
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
