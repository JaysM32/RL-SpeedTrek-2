"""Export TensorBoard scalars (SAC/PPO training curves) to a PNG plot for your report.

Usage:
    python -m scripts.plot_learning_curve --logdir runs/sac_vae --out plots/learning_curve.png

    # plot several tags on one chart, e.g. mean episode reward and mean episode length:
    python -m scripts.plot_learning_curve --logdir runs/sac_vae \
        --tags rollout/ep_rew_mean rollout/ep_len_mean --out plots/reward_and_length.png

    # smooth a noisy curve with a rolling average over 20 data points:
    python -m scripts.plot_learning_curve --logdir runs/sac_vae --smooth 20

    # see every tag that's actually been logged, if you're not sure what to plot:
    python -m scripts.plot_learning_curve --logdir runs/sac_vae --list-tags

Walks every events.out.tfevents.* file found anywhere under --logdir. SB3 creates a new
subfolder each time training starts fresh, and a `--resume` run continues writing into an
existing one -- so after several resumes (phase 1 "safe" mode, phase 2 "fast" mode, etc.)
there's usually more than one event file, and this script merges all of them by step so you
get one continuous curve rather than having to hunt down the right subfolder yourself.

Useful tags SB3 and this project's EpisodeStatsCallback already log, without any extra setup:
    rollout/ep_rew_mean   -- SB3's own rolling average reward over the last 100 episodes
    rollout/ep_len_mean   -- same, for episode length
    rollout/ep_rew        -- this project's own per-episode (non-averaged) reward
    rollout/ep_len        -- same, per-episode length
    train/actor_loss, train/critic_loss, train/ent_coef  -- SAC's internal training losses
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _find_event_files(logdir):
    found = []
    for root, _, files in os.walk(logdir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                found.append(os.path.join(root, f))
    return found


def _load_tag(event_files, tag):
    """Returns (steps, values), sorted by step, merged across every event file that has `tag`."""
    points = []
    for path in event_files:
        ea = EventAccumulator(path, size_guidance={"scalars": 0})  # 0 = no cap, load everything
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for e in ea.Scalars(tag):
            points.append((e.step, e.value))
    points.sort(key=lambda p: p[0])
    if not points:
        return [], []
    steps, values = zip(*points)
    return list(steps), list(values)


def _all_tags(event_files):
    tags = set()
    for path in event_files:
        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        tags.update(ea.Tags().get("scalars", []))
    return sorted(tags)


def _smooth(values, window):
    if window <= 1 or len(values) < 2:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="runs/sac_vae")
    parser.add_argument("--tags", nargs="+", default=["rollout/ep_rew_mean"],
                         help="TensorBoard scalar tags to plot together on one chart")
    parser.add_argument("--out", default="plots/learning_curve.png")
    parser.add_argument("--smooth", type=int, default=1,
                         help="rolling-average window in data points; 1 = no smoothing")
    parser.add_argument("--list-tags", action="store_true",
                         help="print every scalar tag found under --logdir and exit, instead of plotting")
    args = parser.parse_args()

    event_files = _find_event_files(args.logdir)
    if not event_files:
        raise SystemExit(f"no TensorBoard event files found under {args.logdir} -- "
                          f"has training actually logged anything there yet?")
    print(f"found {len(event_files)} event file(s) under {args.logdir}")

    if args.list_tags:
        for tag in _all_tags(event_files):
            print(tag)
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    any_plotted = False
    for tag in args.tags:
        steps, values = _load_tag(event_files, tag)
        if not steps:
            print(f"warning: tag '{tag}' not found in any event file -- skipping "
                  f"(run with --list-tags to see what IS available)")
            continue
        ax.plot(steps, _smooth(values, args.smooth), label=tag)
        any_plotted = True

    if not any_plotted:
        raise SystemExit("none of the requested tags were found -- run with --list-tags to see "
                          "what's actually logged under this logdir")

    ax.set_xlabel("training timestep")
    ax.set_ylabel("value")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved plot to {args.out}")


if __name__ == "__main__":
    main()