"""
Reward shaping for the DonkeyCar env.

gym-donkeycar's `info` dict (from `env.step()`) includes:
    "cte"            -- cross-track error (distance from lane center, signed)
    "speed"          -- current forward speed
    "hit"            -- non-"none" string when a collision/off-track event occurred
    "pos"            -- (x, y, z) world position
    "lap_count"      -- cumulative number of times the car has crossed the start line
    "last_lap_time"  -- time (seconds) of the most recently completed lap

Exact keys can vary slightly by gym-donkeycar version/track -- print `info` once during
your smoke test (see envs/donkey_env.py) and adjust the keys below if needed.

Two curriculum phases, switched via `mode` in configs/sac_vae.yaml's `reward:` section --
no code changes needed to move between them, just edit the yaml and resume training:

  mode: "safe"  (phase 1) -- the only thing that matters is completing the lap without
      leaving the track. Forward-progress reward is capped at `safe_speed_cap` -- moving is
      good, but going faster than that cap earns nothing extra, so speed genuinely doesn't
      matter here. (The cap itself still exists, rather than removing the progress term
      entirely, because a pure "never crash, stay centered" reward with zero incentive to
      move is trivially maximized by sitting still forever -- capping keeps the car actually
      attempting the lap without caring how fast.) Completing a lap earns a flat
      `lap_complete_bonus`, regardless of how long it took.

  mode: "fast"  (phase 2) -- once the agent reliably completes laps in "safe" mode, switch to
      this to start optimizing for speed: the progress reward is no longer capped (faster is
      strictly better), and completing a lap now also earns a bonus that scales with how fast
      that lap was (`lap_time_bonus_scale / last_lap_time` -- shorter lap time, bigger bonus).

Switching modes does NOT touch the VAE or the observation space, so it's safe to `--resume`
an existing SAC checkpoint right after flipping `mode` in the yaml -- unlike retraining the
VAE, which does invalidate a resumed checkpoint (see the VAE track-coverage discussion).
Expect a rough patch for a while after switching as the agent adapts to the new incentive.

Iterate on the weights in configs/sac_vae.yaml once you see how the agent actually behaves --
a policy that crawls slowly to avoid the crash penalty usually means `min_speed_penalty`
needs to be stronger relative to `crash_penalty` (in "safe" mode); a reckless policy usually
means the reverse.
"""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    mode: str = "safe"  # "safe" (phase 1) or "fast" (phase 2) -- see module docstring
    progress_weight: float = 1.0
    cross_track_weight: float = -1.0
    crash_penalty: float = -10.0
    min_speed_penalty: float = -0.5
    min_speed_threshold: float = 1.0
    max_cte: float = 4.0  # |cte| beyond this counts as off-track / episode end
    safe_speed_cap: float = 1.0        # "safe" mode only: progress reward caps out here
    lap_complete_bonus: float = 20.0   # flat bonus on completing a lap, in either mode
    lap_time_bonus_scale: float = 200.0  # "fast" mode only: bonus = this / last_lap_time


def compute_reward(info: dict, cfg: RewardConfig, just_completed_lap: bool = False) -> tuple[float, bool]:
    """Returns (reward, done_due_to_crash).

    `just_completed_lap` must be computed by the caller (see RewardShapingWrapper.step() in
    envs/donkey_env.py) -- a single info dict only has the *cumulative* lap_count, so only
    something that also saw the *previous* step's lap_count can tell whether a crossing just
    happened this step.
    """
    cte = float(info.get("cte", 0.0))
    speed = float(info.get("speed", 0.0))
    hit = info.get("hit", "none")

    crashed = (hit != "none") or (abs(cte) > cfg.max_cte)
    if crashed:
        return cfg.crash_penalty, True

    if cfg.mode == "safe":
        # capped -- moving is good, going *fast* earns nothing extra.
        reward = cfg.progress_weight * max(min(speed, cfg.safe_speed_cap), 0.0)
    else:
        # "fast" mode -- uncapped, faster is strictly better.
        reward = cfg.progress_weight * max(speed, 0.0)

    reward += cfg.cross_track_weight * abs(cte)

    if cfg.mode == "safe" and speed < cfg.min_speed_threshold:
        # only discourage crawling-to-a-stop in "safe" mode -- in "fast" mode the uncapped
        # speed reward above already does this job, and stacking both would double-punish.
        reward += cfg.min_speed_penalty

    if just_completed_lap:
        reward += cfg.lap_complete_bonus
        last_lap_time = float(info.get("last_lap_time", 0.0))
        if cfg.mode == "fast" and last_lap_time > 0:
            reward += cfg.lap_time_bonus_scale / last_lap_time

    return reward, False