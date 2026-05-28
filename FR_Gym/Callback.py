from stable_baselines3.common.callbacks import EvalCallback,CallbackList,BaseCallback,CheckpointCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.n_envs = 0
        self.episode_lengths = []
        self.episode_counts = []
        self.episode_total_rewards = []
        self.episode_dis_rewards = []
        self.episode_joint_success_rewards = []
        self.episode_joint_progress_rewards = []
        self.episode_lagging_distance_penalties = []
        self.episode_incomplete_task_penalties = []
        self.episode_success = []
        self.episode_arm1_success = []
        self.episode_arm2_success = []
        self.episode_robot_collisions = []
        self.episode_final_arm1_distances = []
        self.episode_final_arm2_distances = []

        self.log_interval = 30  # 每30个回合记录一次

    def _on_training_start(self) -> None:
        self.n_envs = self.training_env.num_envs
        self.episode_lengths = [0 for _ in range(self.n_envs)]
        self.episode_counts = [0 for _ in range(self.n_envs)]
        self.episode_total_rewards = [0.0 for _ in range(self.n_envs)]
        self.episode_dis_rewards = [0.0 for _ in range(self.n_envs)]
        self.episode_joint_success_rewards = [0.0 for _ in range(self.n_envs)]
        self.episode_joint_progress_rewards = [0.0 for _ in range(self.n_envs)]
        self.episode_lagging_distance_penalties = [0.0 for _ in range(self.n_envs)]
        self.episode_incomplete_task_penalties = [0.0 for _ in range(self.n_envs)]
        self.episode_success = [0.0 for _ in range(self.n_envs)]
        self.episode_arm1_success = [0.0 for _ in range(self.n_envs)]
        self.episode_arm2_success = [0.0 for _ in range(self.n_envs)]
        self.episode_robot_collisions = [0.0 for _ in range(self.n_envs)]
        self.episode_final_arm1_distances = [0.0 for _ in range(self.n_envs)]
        self.episode_final_arm2_distances = [0.0 for _ in range(self.n_envs)]

    def _on_step(self) -> bool:
        # 遍历所有环境
        for i in range(len(self.locals['rewards'])):
            self.episode_total_rewards[i] += self.locals['rewards'][i]
            self.episode_dis_rewards[i] += self.locals['infos'][i]['distance_reward']
            self.episode_joint_success_rewards[i] += self.locals['infos'][i].get('joint_success_reward', 0)
            self.episode_joint_progress_rewards[i] += self.locals['infos'][i].get('joint_progress_reward', 0)
            self.episode_lagging_distance_penalties[i] += self.locals['infos'][i].get('lagging_distance_penalty', 0)
            self.episode_incomplete_task_penalties[i] += self.locals['infos'][i].get('incomplete_task_penalty', 0)
            self.episode_success[i] += self.locals['infos'][i]['success_reward']
            self.episode_lengths[i] += 1

            # 检查回合是否结束
            if self.locals['dones'][i]:
                self.episode_counts[i] += 1
                self.episode_arm1_success[i] += int(self.locals['infos'][i].get('arm1_success', False))
                self.episode_arm2_success[i] += int(self.locals['infos'][i].get('arm2_success', False))
                self.episode_robot_collisions[i] += int(self.locals['infos'][i].get('robot_collision', False))
                self.episode_final_arm1_distances[i] += self.locals['infos'][i].get('arm1_distance', 0)
                self.episode_final_arm2_distances[i] += self.locals['infos'][i].get('arm2_distance', 0)

                # 每 log_interval 个回合记录一次平均指标
                if self.episode_counts[i] % self.log_interval == 0:
                    avg_reward = self.episode_total_rewards[i] / self.log_interval
                    avg_dis_reward = self.episode_dis_rewards[i] / self.log_interval
                    avg_joint_success_reward = self.episode_joint_success_rewards[i] / self.log_interval
                    avg_joint_progress_reward = self.episode_joint_progress_rewards[i] / self.log_interval
                    avg_lagging_distance_penalty = self.episode_lagging_distance_penalties[i] / self.log_interval
                    avg_incomplete_task_penalty = self.episode_incomplete_task_penalties[i] / self.log_interval
                    avg_success = self.episode_success[i] / self.log_interval
                    avg_arm1_success = self.episode_arm1_success[i] / self.log_interval
                    avg_arm2_success = self.episode_arm2_success[i] / self.log_interval
                    avg_robot_collision = self.episode_robot_collisions[i] / self.log_interval
                    avg_final_arm1_distance = self.episode_final_arm1_distances[i] / self.log_interval
                    avg_final_arm2_distance = self.episode_final_arm2_distances[i] / self.log_interval

                    self.model.logger.record(f"reward/env_{i}", avg_reward, exclude="stdout")
                    self.model.logger.record(f"distance_reward/env_{i}", avg_dis_reward, exclude="stdout")
                    self.model.logger.record(f"joint_success_reward/env_{i}", avg_joint_success_reward, exclude="stdout")
                    self.model.logger.record(f"joint_progress_reward/env_{i}", avg_joint_progress_reward, exclude="stdout")
                    self.model.logger.record(f"lagging_distance_penalty/env_{i}", avg_lagging_distance_penalty, exclude="stdout")
                    self.model.logger.record(f"incomplete_task_penalty/env_{i}", avg_incomplete_task_penalty, exclude="stdout")
                    self.model.logger.record(f"success_rate/env_{i}", avg_success, exclude="stdout")
                    self.model.logger.record(f"arm1_success_rate/env_{i}", avg_arm1_success, exclude="stdout")
                    self.model.logger.record(f"arm2_success_rate/env_{i}", avg_arm2_success, exclude="stdout")
                    self.model.logger.record(f"robot_collision_rate/env_{i}", avg_robot_collision, exclude="stdout")
                    self.model.logger.record(f"final_arm1_distance/env_{i}", avg_final_arm1_distance, exclude="stdout")
                    self.model.logger.record(f"final_arm2_distance/env_{i}", avg_final_arm2_distance, exclude="stdout")

                    self.model.logger.dump(step=self.num_timesteps)

                    # 重置累积奖励和回合长度
                    self.episode_total_rewards[i] = 0.0
                    self.episode_dis_rewards[i] = 0.0
                    self.episode_joint_success_rewards[i] = 0.0
                    self.episode_joint_progress_rewards[i] = 0.0
                    self.episode_lagging_distance_penalties[i] = 0.0
                    self.episode_incomplete_task_penalties[i] = 0.0
                    self.episode_success[i] = 0.0
                    self.episode_arm1_success[i] = 0.0
                    self.episode_arm2_success[i] = 0.0
                    self.episode_robot_collisions[i] = 0.0
                    self.episode_final_arm1_distances[i] = 0.0
                    self.episode_final_arm2_distances[i] = 0.0
                    self.episode_lengths[i] = 0

        return True


class ViewerCallback(BaseCallback):
    def __init__(self, viewer_env, step_freq=1, steps_per_update=1, verbose=0):
        super().__init__(verbose)
        self.viewer_env = viewer_env
        self.step_freq = max(1, int(step_freq))
        self.steps_per_update = max(1, int(steps_per_update))
        self._obs = None

    def _on_training_start(self) -> None:
        self._obs = self.viewer_env.reset()

    def _on_step(self) -> bool:
        if self.n_calls % self.step_freq != 0:
            return True

        dones = [False]
        for _ in range(self.steps_per_update):
            action, _ = self.model.predict(self._obs, deterministic=True)
            self._obs, _, dones, _ = self.viewer_env.step(action)

            if dones[0]:
                self._obs = self.viewer_env.reset()
                break

        self.viewer_env.render()

        return True
