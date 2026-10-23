import argparse
import csv
import math
import os
import time
from datetime import datetime, timezone


def _ground_dist(pos_a, pos_b) -> float:
    return math.hypot(pos_a[0] - pos_b[0], pos_a[2] - pos_b[2])


def run_one_time_trial(env, model, laps: int = 3, deterministic: bool = True,
                        lap_departure_m: float = 2.0, lap_return_m: float = 0.1,
                        out_lap: bool = True, cooldown_lap: bool = True):
    obs, info = env.reset()
    start_pos = info["pos"]
    sim_lap_count_seen = info.get("lap_count", 0)  
    steps = 0
    hit_something = False
    max_dist_from_start = 0.0

    def _drive_to_next_lap_boundary():

        nonlocal obs, steps, max_dist_from_start, sim_lap_count_seen, hit_something
        armed = False
        t0 = time.time()
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            sim_lap_count_seen = max(sim_lap_count_seen, info.get("lap_count", 0))

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
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac",
                         help="which algorithm --model was trained with -- sac expects VAE "
                              "latent observations (needs --vae-checkpoint), ppo expects raw "
                              "resized camera frames (no --vae-checkpoint). Default sac matches "
                              "this script's original behavior.")
    parser.add_argument("--vae-checkpoint", default=None,
                         help="required when --algo sac; ignored for --algo ppo")
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
    parser.add_argument("--max-episode-steps", type=int, default=10000,
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
    parser.add_argument("--frame-skip", type=int, default=2,
                         help="how many simulator physics ticks happen per env.step() call -- "
                              "MUST match whatever the model was actually trained with "
                              "(configs/sac_vae.yaml's env.frame_skip, which is 2 by default) or "
                              "the policy will be making decisions at a different cadence than it "
                              "was trained for. This is purely an environment/evaluation-time "
                              "setting -- it isn't encoded in the saved model weights, so changing "
                              "it here never requires retraining, but it must still match the run "
                              "you're evaluating.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="logs/time_trial.csv",
                         help="CSV file to append every attempt's result to (created if missing)")
    args = parser.parse_args()

    if args.algo == "sac" and not args.vae_checkpoint:
        parser.error("--vae-checkpoint is required for --algo sac")

    if args.algo == "sac":
        from stable_baselines3 import SAC
        from envs.donkey_env import make_vae_env
        model_cls = SAC
        env = make_vae_env(
            sim_path=args.sim_path,
            vae_checkpoint=args.vae_checkpoint,
            env_id=args.env_id,
            max_episode_steps=args.max_episode_steps,
            device=args.device,
            frame_skip=args.frame_skip,
        )
    else:
        from stable_baselines3 import PPO
        from envs.donkey_env import make_raw_donkey_env
        model_cls = PPO
        env = make_raw_donkey_env(
            sim_path=args.sim_path,
            env_id=args.env_id,
            max_episode_steps=args.max_episode_steps,
            frame_skip=args.frame_skip,
        )
    model = model_cls.load(args.model, device=args.device)

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