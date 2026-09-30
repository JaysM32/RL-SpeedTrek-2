"""Turn logs/time_trial.csv (written by scripts/time_trial.py) into report-ready charts.

Usage:
    python -m scripts.plot_time_trial --csv logs/time_trial.csv --out-dir plots

Produces two PNGs:
  1. <out-dir>/time_trial_totals.png  -- total N-lap time per attempt, in the order they were
     run, one line/marker series per model checkpoint (so if you've run this against several
     checkpoints over time -- e.g. an early "fast"-mode checkpoint vs. a later one -- you get
     a visual "did it actually get faster" comparison, not just isolated numbers).
  2. <out-dir>/time_trial_lap_splits.png  -- mean time for lap 1 vs lap 2 vs lap 3 (etc.),
     grouped by checkpoint, using only attempts that actually finished. Useful for seeing
     whether the agent is consistent across laps or fades/speeds up over the run.

Both charts only use FINISHED attempts (a DNF has no total_time or complete lap splits to
plot) -- the finish rate per checkpoint is reported in the chart title/legend and printed to
the console instead, since "how often does it even finish" is a different kind of number than
"how fast is it when it does."
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _lap_columns(rows):
    if not rows:
        return []
    return [k for k in rows[0].keys() if k.startswith("lap_")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/time_trial.csv")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"{args.csv} doesn't exist yet -- run scripts.time_trial at least "
                          f"once first, it creates and appends to this file")

    rows = _load_rows(args.csv)
    if not rows:
        raise SystemExit(f"{args.csv} exists but has no rows yet")

    lap_cols = _lap_columns(rows)
    os.makedirs(args.out_dir, exist_ok=True)

    models = sorted(set(r["model"] for r in rows))
    print(f"found {len(rows)} attempt(s) across {len(models)} checkpoint(s) in {args.csv}")

    # ---- console summary, per checkpoint ----
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        finished = [r for r in model_rows if r["finished"] == "True"]
        print(f"\n{model}")
        print(f"  finish rate: {len(finished)}/{len(model_rows)}")
        if finished:
            totals = [float(r["total_time"]) for r in finished]
            print(f"  best total time: {min(totals):.2f}s   mean: {sum(totals) / len(totals):.2f}s")

    # ---- chart 1: total time per attempt, one series per model ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        finished = [r for r in model_rows if r["finished"] == "True"]
        if not finished:
            continue
        xs = list(range(1, len(finished) + 1))
        ys = [float(r["total_time"]) for r in finished]
        label = f"{os.path.basename(model)} ({len(finished)}/{len(model_rows)} finished)"
        ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel("attempt (finished attempts only, in run order)")
    ax.set_ylabel("total time (s)")
    ax.set_title(f"Time-trial total time ({len(lap_cols)} laps)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    totals_path = os.path.join(args.out_dir, "time_trial_totals.png")
    fig.savefig(totals_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved {totals_path}")

    # ---- chart 2: mean lap split per lap position, grouped by model ----
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_width = 0.8 / max(len(models), 1)
    lap_positions = list(range(len(lap_cols)))

    any_bars = False
    for m_idx, model in enumerate(models):
        finished = [r for r in rows if r["model"] == model and r["finished"] == "True"]
        if not finished:
            continue
        means = []
        for col in lap_cols:
            values = [float(r[col]) for r in finished if r[col] != ""]
            means.append(sum(values) / len(values) if values else 0.0)
        offsets = [p + m_idx * bar_width for p in lap_positions]
        ax.bar(offsets, means, width=bar_width, label=os.path.basename(model))
        any_bars = True

    if any_bars:
        tick_positions = [p + bar_width * (len(models) - 1) / 2 for p in lap_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"lap {i + 1}" for i in lap_positions])
        ax.set_ylabel("mean time (s)")
        ax.set_title("Mean lap split by position (finished attempts only)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "no finished attempts to plot", ha="center", va="center")

    fig.tight_layout()
    splits_path = os.path.join(args.out_dir, "time_trial_lap_splits.png")
    fig.savefig(splits_path, dpi=150)
    plt.close(fig)
    print(f"saved {splits_path}")


if __name__ == "__main__":
    main()