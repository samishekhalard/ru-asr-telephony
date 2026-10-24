"""
Run every engine over every channel and write the results down.

    python -m run --limit 50
    python -m run --limit 50 --channels clean narrowband g711_alaw
    python -m run --limit 5 --engines echo        # wiring check, no model

What this produces
------------------
results/raw/<engine>__<channel>.json   every reference and hypothesis, verbatim
results/summary.md                     the table

The raw transcripts are written because a benchmark whose outputs nobody can
inspect is a claim, not a measurement. Anyone can recompute the rates from
those files, or disagree with the normalisation and apply their own.

The same utterances go through every channel, and every engine sees the same
degraded audio. Only one thing changes at a time.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from data import describe, load_utterances
from degrade import CHANNELS, apply_channel
from engines import build
from metrics import bootstrap_ci, error_rates, paired_delta_ci

RESULTS = Path("results")


def run_one(engine, utterances, channel: str, quiet: bool = False) -> dict:
    """Transcribe every utterance through one channel with one engine."""
    references, hypotheses, timings = [], [], []

    for index, utterance in enumerate(utterances, start=1):
        audio = apply_channel(utterance.audio, utterance.sample_rate, channel)
        result = engine.transcribe(audio, utterance.sample_rate)

        references.append(utterance.reference)
        hypotheses.append(result.text)
        timings.append(result.real_time_factor)

        if not quiet:
            print(
                f"  {engine.name} / {channel}: {index}/{len(utterances)}",
                end="\r",
                flush=True,
            )

    if not quiet:
        print(" " * 60, end="\r")

    rates = error_rates(references, hypotheses)
    lo, hi = bootstrap_ci(references, hypotheses)

    return {
        "engine": engine.name,
        "channel": channel,
        "wer": rates.wer,
        "cer": rates.cer,
        "wer_ci_low": lo,
        "wer_ci_high": hi,
        "n_utterances": rates.n_utterances,
        "n_ref_words": rates.n_ref_words,
        "median_real_time_factor": sorted(timings)[len(timings) // 2],
        "pairs": [
            {"uid": u.uid, "reference": r, "hypothesis": h}
            for u, r, h in zip(utterances, references, hypotheses)
        ],
    }


def delta_table(rows: list[dict], baseline: str = "clean") -> str:
    """
    Each channel against the clean control, measured on the same utterances.

    Independent intervals in the table above overlap easily, because most of
    their width is utterances being unequally hard rather than the channel
    doing anything. Pairing cancels that, so a small consistent effect becomes
    visible. This is the table that answers "does the phone line cost
    anything", and the one above is the table that reports absolute rates.
    """
    by_engine: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_engine.setdefault(row["engine"], {})[row["channel"]] = row

    lines = [
        f"| engine | channel | ΔWER vs {baseline} | 95% CI | verdict |",
        "|---|---|---|---|---|",
    ]
    any_rows = False
    for engine, channels in by_engine.items():
        if baseline not in channels:
            continue
        control = channels[baseline]
        references = [p["reference"] for p in control["pairs"]]
        control_hyps = [p["hypothesis"] for p in control["pairs"]]

        for name in CHANNELS:
            if name == baseline or name not in channels:
                continue
            hyps = [p["hypothesis"] for p in channels[name]["pairs"]]
            delta = paired_delta_ci(references, control_hyps, hyps)
            verdict = "worse" if delta.significant else "not distinguishable"
            lines.append(
                f"| {engine} | {name} | {delta.delta:+.2%} | "
                f"{delta.ci_low:+.2%} – {delta.ci_high:+.2%} | {verdict} |"
            )
            any_rows = True

    return "\n".join(lines) if any_rows else ""


def summary_table(rows: list[dict]) -> str:
    """Markdown, sorted by engine then by how much the channel degrades."""
    order = {name: i for i, name in enumerate(CHANNELS)}
    rows = sorted(rows, key=lambda r: (r["engine"], order.get(r["channel"], 99)))

    lines = [
        "| engine | channel | WER | 95% CI | CER | RTF |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['engine']} | {row['channel']} | {row['wer']:.1%} | "
            f"{row['wer_ci_low']:.1%} – {row['wer_ci_high']:.1%} | "
            f"{row['cer']:.1%} | {row['median_real_time_factor']:.2f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50, help="utterances to use")
    parser.add_argument("--engines", nargs="+", default=["whisper"])
    parser.add_argument("--channels", nargs="+", default=list(CHANNELS))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    for channel in args.channels:
        if channel not in CHANNELS:
            parser.error(f"unknown channel {channel!r}; expected {list(CHANNELS)}")

    print(f"loading {args.limit} utterances...")
    utterances = load_utterances(limit=args.limit)
    print(describe(utterances))
    print()

    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    rows = []

    for engine_name in args.engines:
        engine = build(engine_name)
        for channel in args.channels:
            row = run_one(engine, utterances, channel, quiet=args.quiet)
            rows.append(row)

            safe = row["engine"].replace("/", "_").replace(":", "_")
            path = RESULTS / "raw" / f"{safe}__{channel}.json"
            path.write_text(json.dumps(row, ensure_ascii=False, indent=2))

            print(
                f"{row['engine']:<28} {channel:<12} "
                f"WER {row['wer']:.1%}  "
                f"[{row['wer_ci_low']:.1%}–{row['wer_ci_high']:.1%}]  "
                f"RTF {row['median_real_time_factor']:.2f}"
            )

    table = summary_table(rows)
    deltas = delta_table(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    document = [
        "# Results",
        "",
        f"{describe(utterances)}. Measured {stamp} on "
        f"{platform.machine()} / {platform.system()}.",
        "",
        "## Absolute error rates",
        "",
        table,
        "",
        "RTF is the median real-time factor: below 1.0 is faster than real "
        "time, on one laptop.",
    ]
    if deltas:
        document += [
            "",
            "## Each channel against the clean control",
            "",
            "Paired bootstrap over the same utterances, so differences in "
            "utterance difficulty cancel. A positive delta means the channel "
            "costs accuracy.",
            "",
            deltas,
        ]
    document += ["", "Raw transcripts for every row are in `results/raw/`.", ""]

    RESULTS.joinpath("summary.md").write_text("\n".join(document))

    print()
    print(table)
    if deltas:
        print()
        print(deltas)
    print(f"\nwritten to {RESULTS}/summary.md and {RESULTS}/raw/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
