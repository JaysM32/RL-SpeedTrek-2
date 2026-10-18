"""SB3 callbacks used by the training scripts."""

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


def make_checkpoint_callback(save_freq: int, save_path: str, name_prefix: str) -> CheckpointCallback:
    """Periodic checkpointing so a crashed/interrupted run doesn't lose everything."""
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix=name_prefix,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )


class EpisodeStatsCallback(BaseCallback):
    """Logs episode reward/length, and now lap stats, to TensorBoard as training progresses.

    SB3's Monitor wrapper already records reward/length in `info["episode"]`; this callback
    just surfaces it explicitly so it's easy to find in TensorBoard alongside the SAC/PPO
    losses.

    Lap completions come from info["lap_count_reliable"] / info["last_lap_steps"], which
    envs.donkey_env.RewardShapingWrapper computes itself from the car's position -- NOT from
    gym-donkeycar's own info["lap_count"] / info["last_lap_time"], which undercount laps on
    the waveshare track (see RewardShapingWrapper's docstring) and were what this callback
    used to read. "lap_steps" is a step count, not seconds -- see RewardConfig.lap_steps_
    bonus_scale's docstring for why wall-clock time isn't used during training. Lower
    rollout/lap_steps over training is the "fast" mode curriculum actually working; it isn't
    directly comparable in units to scripts/time_trial.py's second-based lap times, only to
    itself over the course of training.

    Handles vectorized envs (more than one env stepping in parallel) by tracking the last-seen
    lap count *per env index*, not globally -- and resets that per-env tracking the moment an
    episode ends, since a VecEnv auto-resets on episode end and the next info for that env
    index will be a fresh episode starting back at lap_count_reliable=0. Without that reset,
    the first lap of every new episode after the very first one would silently fail to log (0
    is never ">" whatever count the previous episode ended on).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._last_lap_count: dict[int, int] = {}
        self._total_laps_logged = 0

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            ep_info = info.get("episode")
            if ep_info is not None:
                self.logger.record("rollout/ep_rew", ep_info["r"])
                self.logger.record("rollout/ep_len", ep_info["l"])

            lap_count = info.get("lap_count_reliable", 0)
            last_seen = self._last_lap_count.get(i, 0)
            if lap_count > last_seen:
                self._total_laps_logged += 1
                self.logger.record("rollout/lap_steps", float(info.get("last_lap_steps", 0)))
                self.logger.record("rollout/laps_completed_total", self._total_laps_logged)
            self._last_lap_count[i] = lap_count

            if ep_info is not None:
                # this env index just auto-reset -- its next info will be a fresh episode
                # starting back at lap_count_reliable 0, so reset our tracking now rather than
                # comparing the new episode's first lap against the old episode's final count.
                self._last_lap_count[i] = 0

        return True