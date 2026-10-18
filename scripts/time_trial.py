"""Time a trained policy over exactly N laps (default 3) in one continuous run, and report the
total time -- directly comparable to a human's stopwatch time over the same number of laps.

Usage:
    python -m scripts.time_trial --sim-path /path/to/donkey_sim --model checkpoints/sac_donkey.zip \
        --vae-checkpoint checkpoints/vae.pt --laps 3

    # run several attempts and report best + mean, since one attempt can get unlucky:
    python -m scripts.time_trial --sim-path "/Users/jaysm/Desktop/donkey_sim.app/Contents/MacOS/donkey_sim" --model checkpoints/sac_waveshare/sac_donkey_200000_steps.zip --vae-checkpoint checkpoints/vae.pt --laps 3 --attempts 5

    # skip the out/cooldown laps for a quicker sanity-check run:
    python -m scripts.time_trial --sim-path /path/to/donkey_sim --model checkpoints/sac_donkey.zip \
        --vae-checkpoint checkpoints/vae.pt --no-out-lap --no-cooldown-lap
        
        
    python -m scripts.time_trial --sim-path "/Users/jaysm/Desktop/donkey_sim.app/Contents/MacOS/donkey_sim" --model checkpoints/sac_waveshare/sac_donkey_200000_steps.zip --vae-checkpoint checkpoints/vae.pt --laps 3

By default this drives 5 laps total per attempt -- 1 untimed OUT LAP, 3 TIMED LAPS, 1 untimed
COOLDOWN LAP -- the exact same convention scripts/manual_time_trial.py uses for the physical
JetRacer, so a sim result and a real-world result are produced the same way and stay directly
comparable in your report rather than measuring subtly different things. Only the 3 timed laps
count toward total_time/the CSV; the out and cooldown laps exist purely to match the real-world
protocol (a car doesn't start from a dead stop already at race pace, and shouldn't be expected
to brake hard the instant lap 3 ends).

Why not just use scripts/evaluate.py's mean/best single-lap time for the human comparison?
On a simple track a single lap is noisy -- one lucky or unlucky moment swings the result more
than it would over a longer run -- and a human time trial is usually timed over a fixed number
of laps anyway. Summing consecutive individual lap times from one continuous, uninterrupted
run gives the same kind of number a human's stopwatch time over N laps would.

LAP DETECTION: this does NOT use gym-donkeycar's own info["lap_count"]/info["last_lap_time"].
On the waveshare track those undercount real laps -- the sim's lap counter only fires on
`on_collision_with_starting_line`, and only counts a crossing if it hits the exact same
`starting_line_index` as the very first crossing (see gym_donkeycar/envs/donkey_sim.py).
That index isn't exposed to Python at all, so if the start/finish trigger on this track is
built from more than one segment, most physical laps silently don't match and don't count --
which is exactly the "several real laps = one counted lap, then it runs out of steps" symptom.
Instead, this tracks the car's own ground position (info["pos"], Unity is Y-up so the ground
plane is X/Z): once the car has moved at least --lap-departure-m away from its start point
(so it's genuinely left, not idling near the line), the next time it comes back within
--lap-return-m of that start point counts as one lap, timed with a Python-side stopwatch from
the previous lap boundary. This is independent of whatever the simulator's own start-line
trigger geometry happens to be for a given track.

Note: --max-episode-steps defaults much higher than configs/sac_vae.yaml's training value
(2000, sized for roughly one lap) -- sized for 5 laps (out + 3 timed + cooldown) by default,
not just --laps, otherwise an attempt would frequently get cut off by the episode step limit
before finishing, which would look like a crash/DNF when it isn't.

Every attempt is also appended to a CSV (--out, default logs/time_trial.csv) -- one row per
attempt, with the model checkpoint path, timestamp, per-lap splits, total time, and DNF reason
if it didn't finish. Console output alone disappears when the terminal closes; this is what
you'd actually cite a number from in a report, and what lets you compare results across
different checkpoints later without having to re-run anything.
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime, timezone

from envs.donkey_env import make_vae_env


def _ground_dist(pos_a, pos_b) -> float:
    """Distance in the ground plane only. gym-donkeycar's Unity scenes are Y-up, so "ground
    plane" means X/Z -- Y is height, and a flat oval track shouldn't need it (if your car is
    somehow airborne this will be wrong, but that's a bigger problem than lap counting)."""
    return math.hypot(pos_a[0] - pos_b[0], pos_a[2] - pos_b[2])


def run_one_time_trial(env, model, laps: int = 3, deterministic: bool = True,
                        lap_departure_m: float = 5.0, lap_return_m: float = 2.0,
                        out_lap: bool = True, cooldown_lap: bool = True):
    """Drives one continuous run: an optional untimed OUT LAP, then `laps` TIMED LAPS, then an
    optional untimed COOLDOWN LAP -- 5 physical laps by default (1 + 3 + 1), matching the same
    convention scripts/manual_time_trial.py uses on the real JetRacer. A crash or running out
    of the step budget during the out lap or the timed laps fails the whole attempt (DNF); a
    crash/truncation during the cooldown lap does NOT -- the timed portion already succeeded,
    and real time trials don't retroactively erase your lap times because of what happened
    after you crossed the final timed line. Doesn't reset the episode partway through, so a
    slow/cautious lap doesn't invalidate the run, only a real crash or running out of steps does.

    Lap boundaries are detected from the car's own position relative to its start point (see
    module docstring for why -- the sim's own lap_count is unreliable on this track), not from
    info["lap_count"]. --lap-departure-m/--lap-return-m are in whatever units info["pos"] uses
    (meters, for gym-donkeycar); defaults are a starting guess, not a verified value for your
    track -- the result dict's "max_dist_from_start" tells you how far the car actually got
    from its start point during the run, which is the quickest way to sanity-check whether
    these thresholds are in the right ballpark for your oval's actual size.
    """
    obs, info = env.reset()
    start_pos = info["pos"]
    sim_lap_count_seen = info.get("lap_count", 0)  # logged only for comparison, see below
    steps = 0
    hit_something = False
    max_dist_from_start = 0.0

    def _drive_to_next_lap_boundary():
        """Drives until the car has left the start area and come back (one lap boundary), or
        the episode ends first. Returns (lap_time_seconds_or_None, ended_episode: bool)."""
        nonlocal obs, steps, max_dist_from_start, sim_lap_count_seen, hit_something
        armed = False
        t0 = time.time()
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            sim_lap_count_seen = max(sim_lap_count_seen, info.get("lap_count", 0))

            # Check termination BEFORE the lap-boundary condition: if the car crashes on the
            # exact same step it also happens to be back near the start point (plausible right
            # at the start/finish line itself), that's a crash, not a quietly-completed lap --
            # checking lap-boundary first would mask the crash entirely.
            if terminated or truncated:
                hit_something = info.get("hit", "none") != "none"
                return None, True

            dist_to_start = _ground_dist(info["pos"], start_pos)
            max_dist_from_start = max(max_dist_from_start, dist_to_start)
            if not armed:
                if dist_to_start > lap_departure_m:
                    armed = True
            elif dist_to_start < lap_return_m:
                return time.time() - t0, False

    out_lap_time = None
    if out_lap:
        out_lap_time, ended = _drive_to_next_lap_boundary()
        if ended:
            return {
                "finished": False, "lap_times": [], "total_time": None, "steps": steps,
                "crashed_before_finish": hit_something, "max_dist_from_start": max_dist_from_start,
                "sim_lap_count_seen": sim_lap_count_seen, "out_lap_time": None,
                "cooldown_lap_time": None, "dnf_phase": "out_lap",
            }

    lap_times = []
    for _ in range(laps):
        lap_time, ended = _drive_to_next_lap_boundary()
        if ended:
            break
        lap_times.append(lap_time)

    finished = len(lap_times) >= laps

    # The timed portion is done -- a cooldown lap is a courtesy, not part of the measurement,
    # so whatever happens during it doesn't change `finished` or the lap times already recorded.
    cooldown_lap_time = None
    if finished and cooldown_lap:
        cooldown_lap_time, _ended = _drive_to_next_lap_boundary()

    return {
        "finished": finished,
        "lap_times": lap_times,
        "total_time": sum(lap_times) if finished else None,
        "steps": steps,
        "crashed_before_finish": (not finished) and hit_something,
        "max_dist_from_start": max_dist_from_start,
        "sim_lap_count_seen": sim_lap_count_seen,
        "out_lap_time": out_lap_time,
        "cooldown_lap_time": cooldown_lap_time,
        "dnf_phase": None if finished else "timed_laps",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--env-id", default="donkey-waveshare-v0")
    parser.add_argument("--laps", type=int, default=3,
                         help="number of TIMED laps (the out lap and cooldown lap, if enabled, "
                              "are extra and untimed -- see --out-lap/--cooldown-lap)")
    parser.add_argument("--out-lap", dest="out_lap", action="store_true", default=True,
                         help="drive one untimed lap before timing starts (default: on)")
    parser.add_argument("--no-out-lap", dest="out_lap", action="store_false",
                         help="skip the untimed out lap -- timing starts immediately")
    parser.add_argument("--cooldown-lap", dest="cooldown_lap", action="store_true", default=True,
                         help="drive one untimed lap after the timed laps finish (default: on)")
    parser.add_argument("--no-cooldown-lap", dest="cooldown_lap", action="store_false",
                         help="skip the untimed cooldown lap -- stop as soon as timing ends")
    parser.add_argument("--attempts", type=int, default=5,
                         help="separate time-trial attempts to run (reports best + mean across them)")
    parser.add_argument("--max-episode-steps", type=int, default=20000,
                         help="step budget for ONE time trial -- must be generous enough for the "
                              "whole attempt, not just --laps: by default that's 5 physical laps "
                              "(1 out + --laps timed + 1 cooldown), not just the timed ones; "
                              "training's 2000-step budget is sized for ~1 lap")
    parser.add_argument("--lap-departure-m", type=float, default=2.0,
                         help="how far (in info['pos'] units) the car must get from its start "
                              "point before a return counts as a lap -- see module docstring. "
                              "Unverified default; check the printed max_dist_from_start on your "
                              "first run to see if this is in the right ballpark for your track.")
    parser.add_argument("--lap-return-m", type=float, default=0.1,
                         help="how close (in info['pos'] units) the car must get back to its "
                              "start point to count as completing a lap")
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
        result = run_one_time_trial(
            env, model, args.laps,
            lap_departure_m=args.lap_departure_m, lap_return_m=args.lap_return_m,
            out_lap=args.out_lap, cooldown_lap=args.cooldown_lap,
        )
        results.append(result)
        if result["finished"]:
            lap_str = ", ".join(f"{t:.2f}s" for t in result["lap_times"])
            extra = []
            if result["out_lap_time"] is not None:
                extra.append(f"out lap {result['out_lap_time']:.2f}s")
            if result["cooldown_lap_time"] is not None:
                extra.append(f"cooldown {result['cooldown_lap_time']:.2f}s")
            extra_str = f"  ({', '.join(extra)}, neither counted)" if extra else ""
            print(f"attempt {attempt}: FINISHED {args.laps} timed laps in {result['total_time']:.2f}s  "
                  f"(laps: {lap_str}){extra_str}")
        else:
            reason = "crashed" if result["crashed_before_finish"] else "ran out of steps"
            phase = "during the out lap" if result["dnf_phase"] == "out_lap" else "during the timed laps"
            print(f"attempt {attempt}: DID NOT FINISH -- {reason} {phase}, after "
                  f"{len(result['lap_times'])}/{args.laps} timed laps")
        # diagnostic only, not persisted to the CSV -- compares this run's position-based lap
        # count against the simulator's own (on this track, unreliable) info["lap_count"], and
        # reports how far the car actually got from its start point so you can tell whether
        # --lap-departure-m/--lap-return-m are sized sensibly for this track's real scale.
        print(f"    [diagnostic] max distance reached from start point: "
              f"{result['max_dist_from_start']:.2f} (vs. --lap-departure-m={args.lap_departure_m})  "
              f"sim's own info['lap_count'] reached: {result['sim_lap_count_seen']} "
              f"(position-based timed laps counted: {len(result['lap_times'])})")

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