from collections import deque

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


def make_checkpoint_callback(save_freq: int, save_path: str, name_prefix: str) -> CheckpointCallback:
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix=name_prefix,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )


class EpisodeStatsCallback(BaseCallback):
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
                self._last_lap_count[i] = 0

        return True


class BestModelCallback(BaseCallback):
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