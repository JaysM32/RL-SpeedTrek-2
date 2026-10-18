from dataclasses import dataclass


@dataclass
class RewardConfig:
    mode: str = "safe"
    progress_weight: float = 1.0
    cross_track_weight: float = -1.0
    crash_penalty: float = -10.0
    min_speed_penalty: float = -0.5
    min_speed_threshold: float = 1.0
    max_cte: float = 4.0
    safe_speed_cap: float = 1.0
    lap_complete_bonus: float = 20.0
    # Scales the "fast" mode speed-of-lap bonus: reward += lap_steps_bonus_scale / lap_steps
    # on the step a lap is completed. This used to be lap_time_bonus_scale / last_lap_time
    # (wall-clock seconds, read from gym-donkeycar's own info["last_lap_time"]) -- changed to
    # steps for two reasons: (1) info["last_lap_time"] only ever got populated when the sim's
    # own on_collision_with_starting_line() recognized the crossing, which on the waveshare
    # track silently fails most of the time (see envs/donkey_env.py's RewardShapingWrapper
    # docstring) -- the bonus was rarely firing at all; (2) even when it did fire, wall-clock
    # seconds measured during training are contaminated by however long gradient updates/model
    # inference took between env.step() calls, which has nothing to do with how fast the car
    # actually drove. Step count isn't contaminated by that and is what "fast" mode should
    # actually be minimizing. Default is a rough starting point, not tuned against a real run
    # -- watch rollout/lap_steps in TensorBoard once you have real episodes and adjust so this
    # bonus is a meaningful (but not dominant) fraction of a typical episode's total reward.
    lap_steps_bonus_scale: float = 5000.0
    # Position-based lap-boundary detection (see RewardShapingWrapper): how far (in
    # info["pos"] units, meters) the car must get from its reset() position before a return
    # counts as a lap, and how close it must come back. Same defaults as
    # scripts/time_trial.py's --lap-departure-m/--lap-return-m, deliberately -- keeping them
    # equal means training and evaluation agree on what a "lap" is on this track.
    lap_departure_m: float = 5.0
    lap_return_m: float = 2.0


def compute_reward(
    info: dict,
    cfg: RewardConfig,
    just_completed_lap: bool = False,
    lap_steps: int | None = None,
) -> tuple[float, bool]:
    cte = float(info.get("cte", 0.0))
    speed = float(info.get("speed", 0.0))
    hit = info.get("hit", "none")

    crashed = (hit != "none") or (abs(cte) > cfg.max_cte)
    if crashed:
        return cfg.crash_penalty, True

    if cfg.mode == "safe":
        reward = cfg.progress_weight * max(min(speed, cfg.safe_speed_cap), 0.0)
    else:
        reward = cfg.progress_weight * max(speed, 0.0)

    reward += cfg.cross_track_weight * abs(cte)

    if cfg.mode == "safe" and speed < cfg.min_speed_threshold:
        reward += cfg.min_speed_penalty

    if just_completed_lap:
        reward += cfg.lap_complete_bonus
        if cfg.mode == "fast" and lap_steps is not None and lap_steps > 0:
            reward += cfg.lap_steps_bonus_scale / lap_steps

    return reward, False