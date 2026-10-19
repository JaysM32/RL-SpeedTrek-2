"""SB3 callbacks used by the training scripts."""

from collections import deque

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


class BestModelCallback(BaseCallback):
    """Saves the model to `save_path` whenever a rolling mean of recent episode rewards
    improves on the best seen so far during THIS training invocation.

    This is a deliberately lighter-weight stand-in for SB3's own EvalCallback. A real
    EvalCallback needs a second, separate environment for periodic deterministic evaluation
    episodes -- for gym-donkeycar that means a second live Unity simulator process, which
    isn't practical when you've only got one donkey_sim.app instance (and one GPU/CPU budget)
    on your machine. This callback costs nothing extra: it just watches the SAME episode
    rewards the training env is already producing (via SB3's Monitor wrapper, the same
    info["episode"]["r"] EpisodeStatsCallback logs), smoothed over a rolling window so one
    lucky/unlucky episode doesn't trigger a save. The tradeoff is that "best" is judged on the
    (stochastic) training policy's own episodes rather than clean deterministic eval runs --
    good enough to avoid walking away with a final checkpoint that's worse than one you drove
    through earlier, which is the actual problem this solves.

    IMPORTANT: "best" is only meaningful WITHIN one invocation / one reward mode. Safe-mode
    and fast-mode rewards are on different scales (fast adds an unbounded speed term and a
    lap-steps bonus) -- but since this callback's _best_mean_reward always starts at -inf
    fresh at the start of each training invocation, a safe-mode run's "best" and a later
    fast-mode run's "best" are never compared against each other; just don't eyeball the two
    numbers side by side and read anything into which is bigger.

    Does NOT help a run that already finished before this callback existed -- for that, see
    scripts/pick_best_checkpoint.py, which re-evaluates the periodic checkpoints
    CheckpointCallback already saved during that run.
    """

    def __init__(self, save_path: str, window_size: int = 20, min_episodes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.window_size = window_size
        self.min_episodes = min_episodes
        self._rewards: dict[int, deque] = {}
        self._best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            ep_info = info.get("episode")
            if ep_info is None:
                continue
            if i not in self._rewards:
                self._rewards[i] = deque(maxlen=self.window_size)
            self._rewards[i].append(ep_info["r"])

        all_recent = [r for buf in self._rewards.values() for r in buf]
        if len(all_recent) < self.min_episodes:
            return True

        mean_reward = sum(all_recent) / len(all_recent)
        if mean_reward > self._best_mean_reward:
            self._best_mean_reward = mean_reward
            self.model.save(self.save_path)
            self.logger.record("rollout/best_mean_ep_rew", mean_reward)
            if self.verbose:
                print(f"[BestModelCallback] new best rolling mean reward {mean_reward:.2f} "
                      f"(over last {len(all_recent)} episode(s)) -- saved to {self.save_path}",
                      flush=True)

        return True