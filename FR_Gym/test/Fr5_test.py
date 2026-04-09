'''
 @Author: Prince Wang 
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''
import os
import time
import numpy as np

from stable_baselines3 import PPO

from ..Fr5_env import FR5_Env
from utils.arguments import get_args

if __name__ == '__main__':
    args, kwargs = get_args()
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    os.makedirs(args.test, exist_ok=True)

    env = FR5_Env(gui=args.gui)
    if args.gui:
        env.render()
    model = PPO.load(args.model_path)
    test_num = args.test_num  # 测试次数
    success_num = 0  # 成功次数
    print("测试模型：", args.model_path)
    print("测试次数：", test_num)
    last_arm1_traj = []
    last_arm2_traj = []
    last_episode_info = None
    for i in range(test_num):
        state, _ = env.reset()
        done = False 
        score = 0
        arm1_traj = []
        arm2_traj = []

        while not done:
            snapshot = env.get_debug_snapshot()
            arm1_traj.append(snapshot["arm1_gripper_center"].copy())
            arm2_traj.append(snapshot["arm2_gripper_center"].copy())
            action, _ = model.predict(observation=state, deterministic=True)
            state, reward, done, _, info = env.step(action=action)
            score += reward
            if args.gui:
                time.sleep(0.01)

        snapshot = env.get_debug_snapshot()
        arm1_traj.append(snapshot["arm1_gripper_center"].copy())
        arm2_traj.append(snapshot["arm2_gripper_center"].copy())

        if info['is_success']:
            success_num += 1
        print(f"第{i + 1}次测试: reward={score:.3f}, success={info['is_success']}")
        last_arm1_traj = arm1_traj
        last_arm2_traj = arm2_traj
        last_episode_info = info

    success_rate = success_num / test_num
    print("成功率：", success_rate)

    trajectory_path = os.path.join(args.test, "last_test_trajectory.npz")
    snapshot = env.get_debug_snapshot()
    np.savez(
        trajectory_path,
        arm1_traj=np.asarray(last_arm1_traj, dtype=np.float32),
        arm2_traj=np.asarray(last_arm2_traj, dtype=np.float32),
        target_1=snapshot["target_1"],
        target_2=snapshot["target_2"],
        success=bool(last_episode_info["is_success"]) if last_episode_info is not None else False,
        step_num=int(last_episode_info["step_num"]) if last_episode_info is not None else 0,
    )
    print("最后一次测试轨迹已保存：", trajectory_path)

    if args.gui and last_arm1_traj and last_arm2_traj:
        env.draw_trajectory(last_arm1_traj, last_arm2_traj)
        print("GUI 已绘制最后一次测试轨迹，关闭窗口前可持续查看。")
        time.sleep(10)

    env.close()
