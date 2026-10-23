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
    lap_steps_bonus_scale: float = 5000.0
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