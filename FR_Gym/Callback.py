from stable_baselines3.common.callbacks import EvalCallback,CallbackList,BaseCallback,CheckpointCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.n_envs = 0
        self.episode_lengths = []
        self.episode_counts = []
        self.episode_total_rewards = []
        self.episode_dis_rewards = []
        self.episode_success = []

        self.log_interval = 30  # 每30个回合记录一次

    def _on_training_start(self) -> None:
        self.n_envs = self.training_env.num_envs
        self.episode_lengths = [0 for _ in range(self.n_envs)]
        self.episode_counts = [0 for _ in range(self.n_envs)]
        self.episode_total_rewards = [0.0 for _ in range(self.n_envs)]
        self.episode_dis_rewards = [0.0 for _ in range(self.n_envs)]
        self.episode_success = [0.0 for _ in range(self.n_envs)]

    def _on_step(self) -> bool:
        # 遍历所有环境
        for i in range(len(self.locals['rewards'])):
            self.episode_total_rewards[i] += self.locals['rewards'][i]
            self.episode_dis_rewards[i] += self.locals['infos'][i]['distance_reward']
            self.episode_success[i] += self.locals['infos'][i]['success_reward']
            self.episode_lengths[i] += 1

            # 检查回合是否结束
            if self.locals['dones'][i]:
                self.episode_counts[i] += 1

                # 每 log_interval 个回合记录一次平均指标
                if self.episode_counts[i] % self.log_interval == 0:
                    avg_reward = self.episode_total_rewards[i] / self.log_interval
                    avg_dis_reward = self.episode_dis_rewards[i] / self.log_interval
                    avg_success = self.episode_success[i] / self.log_interval

                    self.model.logger.record(f"reward/env_{i}", avg_reward, exclude="stdout")
                    self.model.logger.record(f"distance_reward/env_{i}", avg_dis_reward, exclude="stdout")
                    self.model.logger.record(f"success_rate/env_{i}", avg_success, exclude="stdout")

                    self.model.logger.dump(step=self.num_timesteps)

                    # 重置累积奖励和回合长度
                    self.episode_total_rewards[i] = 0.0
                    self.episode_dis_rewards[i] = 0.0
                    self.episode_success[i] = 0.0
                    self.episode_lengths[i] = 0

        return True


class ViewerCallback(BaseCallback):
    def __init__(self, viewer_env, step_freq=20, verbose=0):
        super().__init__(verbose)
        self.viewer_env = viewer_env
        self.step_freq = step_freq
        self._obs = None

    def _on_training_start(self) -> None:
        self._obs = self.viewer_env.reset()

    def _on_step(self) -> bool:
        if self.n_calls % self.step_freq != 0:
            return True

        action, _ = self.model.predict(self._obs, deterministic=True)
        self._obs, _, dones, _ = self.viewer_env.step(action)
        self.viewer_env.render()

        if dones[0]:
            self._obs = self.viewer_env.reset()

        return True
