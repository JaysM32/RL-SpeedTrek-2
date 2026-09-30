"""Time a trained policy over exactly N laps (default 3) in one continuous run, and report the
total time -- directly comparable to a human's stopwatch time over the same number of laps.

Usage:
    python -m scripts.time_trial --sim-path /path/to/donkey_sim --model checkpoints/sac_donkey.zip \
        --vae-checkpoint checkpoints/vae.pt --laps 3

    # run several attempts and report best + mean, since one attempt can get unlucky:
    python -m scripts.time_trial --sim-path /path/to/donkey_sim --model checkpoints/sac_donkey.zip \
        --vae-checkpoint checkpoints/vae.pt --laps 3 --attempts 5

Why not just use scripts/evaluate.py's mean/best single-lap time for the human comparison?
On a simple track a single lap is noisy -- one lucky or unlucky moment swings the result more
than it would over a longer run -- and a human time trial is usually timed over a fixed number
of laps anyway. Summing consecutive individual lap times from one continuous, uninterrupted
run gives the same kind of number a human's stopwatch time over N laps would.

Note: --max-episode-steps defaults much higher than configs/sac_vae.yaml's training value
(2000, sized for roughly one lap) -- otherwise a 3-lap attempt would frequently get cut off by
the episode step limit before finishing lap 3, which would look like a crash/DNF when it isn't.

Every attempt is also appended to a CSV (--out, default logs/time_trial.csv) -- one row per
attempt, with the model checkpoint path, timestamp, per-lap splits, total time, and DNF reason
if it didn't finish. Console output alone disappears when the terminal closes; this is what
you'd actually cite a number from in a report, and what lets you compare results across
different checkpoints later without having to re-run anything.
"""

import argparse
import csv
import os
from datetime import datetime, timezone

from envs.donkey_env import make_vae_env


def run_one_time_trial(env, model, laps: int, deterministic: bool = True):
    """Drives one continuous run until `laps` laps are completed, a crash ends it early, or the
    step budget runs out. Returns a dict describing what happened -- doesn't reset the episode
    partway through, so a slow/cautious lap 1 doesn't invalidate the run, only a real crash or
    running out of steps does.
    """
    obs, info = env.reset()
    lap_times = []
    last_lap_count = 0
    steps = 0
    hit_something = False

    while len(lap_times) < laps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        lap_count = info.get("lap_count", 0)
        if lap_count > last_lap_count:
            lap_times.append(info.get("last_lap_time", 0.0))
            last_lap_count = lap_count

        if terminated or truncated:
            hit_something = info.get("hit", "none") != "none"
            break

    finished = len(lap_times) >= laps
    return {
        "finished": finished,
        "lap_times": lap_times,
        "total_time": sum(lap_times) if finished else None,
        "steps": steps,
        "crashed_before_finish": (not finished) and hit_something,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--env-id", default="donkey-circuit-launch-track-v0")
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=5,
                         help="separate time-trial attempts to run (reports best + mean across them)")
    parser.add_argument("--max-episode-steps", type=int, default=6000,
                         help="step budget for ONE time trial -- must be generous enough for "
                              "--laps laps; training's 2000-step budget is sized for ~1 lap")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="logs/time_trial.csv",
                         help="CSV file to append every attempt's result to (created if missing)")
    args = parser.parse_args()

    from stable_baselines3 import SAC

    env = make_vae_env(
        sim_path=args.sim_path,
        vae_checkpoint=args.vae_checkpoint,
        env_id=args.env_id,
        max_episode_steps=args.max_episode_steps,
        device=args.device,
    )
    model = SAC.load(args.model, device=args.device)

    results = []
    for attempt in range(1, args.attempts + 1):
        result = run_one_time_trial(env, model, args.laps)
        results.append(result)
        if result["finished"]:
            lap_str = ", ".join(f"{t:.2f}s" for t in result["lap_times"])
            print(f"attempt {attempt}: FINISHED {args.laps} laps in {result['total_time']:.2f}s  "
                  f"(laps: {lap_str})")
        else:
            reason = "crashed" if result["crashed_before_finish"] else "ran out of steps"
            print(f"attempt {attempt}: DID NOT FINISH -- {reason} after "
                  f"{len(result['lap_times'])}/{args.laps} laps")

    env.close()

    finished_runs = [r for r in results if r["finished"]]
    print("\n--- summary ---")
    print(f"finished {len(finished_runs)}/{args.attempts} attempts")
    if finished_runs:
        totals = [r["total_time"] for r in finished_runs]
        best = min(totals)
        mean = sum(totals) / len(totals)
        print(f"best {args.laps}-lap time: {best:.2f}s")
        print(f"mean {args.laps}-lap time (successful attempts only): {mean:.2f}s")
        print(f"\ncompare this directly against a human's stopwatch time over the same "
              f"{args.laps} laps on the same track.")
    else:
        print(f"no attempt completed all {args.laps} laps -- the policy isn't reliable enough "
              f"yet for a fair time-trial comparison; keep training 'fast' mode, or pass "
              f"--laps 1 or 2 to see how far it currently gets.")

    _save_csv(args, results)


def _save_csv(args, results):
    """Appends one row per attempt to --out, creating it (with a header) if it doesn't exist
    yet. Appending rather than overwriting means running this script again later -- against
    the same checkpoint to gather more attempts, or against a newer checkpoint -- builds up a
    single running history instead of each run clobbering the last one's results."""
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    file_exists = os.path.exists(args.out)

    lap_columns = [f"lap_{i + 1}" for i in range(args.laps)]
    header = ["timestamp", "model", "env_id", "laps_target", "attempt", "finished",
              "total_time", "dnf_reason", "steps"] + lap_columns

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        for attempt_num, result in enumerate(results, start=1):
            if result["finished"]:
                dnf_reason = ""
            else:
                dnf_reason = "crashed" if result["crashed_before_finish"] else "out_of_steps"

            lap_values = [f"{t:.3f}" for t in result["lap_times"]]
            lap_values += [""] * (args.laps - len(lap_values))  # pad DNF rows to a fixed width

            row = [
                timestamp,
                args.model,
                args.env_id,
                args.laps,
                attempt_num,
                result["finished"],
                f"{result['total_time']:.3f}" if result["total_time"] is not None else "",
                dnf_reason,
                result["steps"],
            ] + lap_values
            writer.writerow(row)

    print(f"\nappended {len(results)} attempt(s) to {args.out}")


if __name__ == "__main__":
    main()