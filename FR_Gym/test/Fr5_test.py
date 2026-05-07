'''
 @Author: Prince Wang 
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from stable_baselines3 import PPO
import sys

if __package__ in (None, ""):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from FR_Gym.Fr5_env import FR5_Env
else:
    from ..Fr5_env import FR5_Env
from utils.arguments import get_args


def save_trajectory_plots(arm1_traj, arm2_traj, target_1, target_2, output_dir, success_distance):
    arm1 = np.asarray(arm1_traj, dtype=np.float32)
    arm2 = np.asarray(arm2_traj, dtype=np.float32)
    target_1 = np.asarray(target_1, dtype=np.float32)
    target_2 = np.asarray(target_2, dtype=np.float32)

    plot_2d_path = os.path.join(output_dir, "last_test_trajectory_2d.png")
    plot_3d_path = os.path.join(output_dir, "last_test_trajectory_3d.png")
    all_points = np.vstack([arm1, arm2, target_1[None, :], target_2[None, :]])

    def expand_limits(values, pad_ratio=0.15, min_span=0.1):
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        span = max(vmax - vmin, min_span)
        pad = span * pad_ratio
        center = (vmin + vmax) / 2.0
        half = span / 2.0 + pad
        return center - half, center + half

    xlim = expand_limits(all_points[:, 0])
    ylim = expand_limits(all_points[:, 1])
    zlim = expand_limits(all_points[:, 2], min_span=0.08)

    fig2d, ax2d = plt.subplots(figsize=(8, 6))
    ax2d.plot(arm1[:, 0], arm1[:, 1], color="red", label="arm1 traj")
    ax2d.plot(arm2[:, 0], arm2[:, 1], color="blue", label="arm2 traj")
    ax2d.scatter(arm1[0, 0], arm1[0, 1], color="red", marker="o", s=40, label="arm1 start")
    ax2d.scatter(arm2[0, 0], arm2[0, 1], color="blue", marker="o", s=40, label="arm2 start")
    ax2d.scatter(arm1[-1, 0], arm1[-1, 1], color="red", marker="s", s=40, label="arm1 end")
    ax2d.scatter(arm2[-1, 0], arm2[-1, 1], color="blue", marker="s", s=40, label="arm2 end")
    ax2d.scatter(target_1[0], target_1[1], color="darkred", marker="x", s=80, label="target1")
    ax2d.scatter(target_2[0], target_2[1], color="navy", marker="x", s=80, label="target2")
    ax2d.add_patch(Circle((target_1[0], target_1[1]), success_distance, color="darkred", alpha=0.12))
    ax2d.add_patch(Circle((target_2[0], target_2[1]), success_distance, color="navy", alpha=0.12))
    ax2d.set_title("Last Test Trajectory 2D (Top View)")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax2d.set_xlim(*xlim)
    ax2d.set_ylim(*ylim)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.legend()
    ax2d.grid(True, alpha=0.3)
    fig2d.tight_layout()
    fig2d.savefig(plot_2d_path, dpi=200)
    plt.close(fig2d)

    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.plot(arm1[:, 0], arm1[:, 1], arm1[:, 2], color="red", label="arm1 traj")
    ax3d.plot(arm2[:, 0], arm2[:, 1], arm2[:, 2], color="blue", label="arm2 traj")
    ax3d.scatter(arm1[0, 0], arm1[0, 1], arm1[0, 2], color="red", marker="o", s=40, label="arm1 start")
    ax3d.scatter(arm2[0, 0], arm2[0, 1], arm2[0, 2], color="blue", marker="o", s=40, label="arm2 start")
    ax3d.scatter(arm1[-1, 0], arm1[-1, 1], arm1[-1, 2], color="red", marker="s", s=40, label="arm1 end")
    ax3d.scatter(arm2[-1, 0], arm2[-1, 1], arm2[-1, 2], color="blue", marker="s", s=40, label="arm2 end")
    ax3d.scatter(target_1[0], target_1[1], target_1[2], color="darkred", marker="x", s=80, label="target1")
    ax3d.scatter(target_2[0], target_2[1], target_2[2], color="navy", marker="x", s=80, label="target2")
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 16)
    sphere_x_1 = target_1[0] + success_distance * np.outer(np.cos(u), np.sin(v))
    sphere_y_1 = target_1[1] + success_distance * np.outer(np.sin(u), np.sin(v))
    sphere_z_1 = target_1[2] + success_distance * np.outer(np.ones_like(u), np.cos(v))
    sphere_x_2 = target_2[0] + success_distance * np.outer(np.cos(u), np.sin(v))
    sphere_y_2 = target_2[1] + success_distance * np.outer(np.sin(u), np.sin(v))
    sphere_z_2 = target_2[2] + success_distance * np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_wireframe(sphere_x_1, sphere_y_1, sphere_z_1, color="darkred", alpha=0.15, linewidth=0.5)
    ax3d.plot_wireframe(sphere_x_2, sphere_y_2, sphere_z_2, color="navy", alpha=0.15, linewidth=0.5)
    ax3d.set_title("Last Test Trajectory 3D")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_xlim(*xlim)
    ax3d.set_ylim(*ylim)
    ax3d.set_zlim(*zlim)
    ax3d.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax3d.view_init(elev=25, azim=-60)
    ax3d.legend()
    fig3d.tight_layout()
    fig3d.savefig(plot_3d_path, dpi=200)
    plt.close(fig3d)

    return plot_2d_path, plot_3d_path

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
    selected_arm1_traj = []
    selected_arm2_traj = []
    selected_episode_info = None
    selected_label = "最后一次测试"
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
        if info['is_success']:
            selected_arm1_traj = arm1_traj
            selected_arm2_traj = arm2_traj
            selected_episode_info = info
            selected_label = f"最后一次成功测试(第{i + 1}次)"

    success_rate = success_num / test_num
    print("成功率：", success_rate)

    if not selected_arm1_traj:
        selected_arm1_traj = last_arm1_traj
        selected_arm2_traj = last_arm2_traj
        selected_episode_info = last_episode_info
        selected_label = "最后一次测试(未成功，回退保存)"

    trajectory_path = os.path.join(args.test, "last_test_trajectory.npz")
    snapshot = env.get_debug_snapshot()
    np.savez(
        trajectory_path,
        arm1_traj=np.asarray(selected_arm1_traj, dtype=np.float32),
        arm2_traj=np.asarray(selected_arm2_traj, dtype=np.float32),
        target_1=snapshot["target_1"],
        target_2=snapshot["target_2"],
        success=bool(selected_episode_info["is_success"]) if selected_episode_info is not None else False,
        step_num=int(selected_episode_info["step_num"]) if selected_episode_info is not None else 0,
    )
    print("最后一次测试轨迹已保存：", trajectory_path)
    print("保存轨迹来源：", selected_label)

    plot_2d_path, plot_3d_path = save_trajectory_plots(
        selected_arm1_traj,
        selected_arm2_traj,
        snapshot["target_1"],
        snapshot["target_2"],
        args.test,
        env.success_distance,
    )
    print("2D轨迹图已保存：", plot_2d_path)
    print("3D轨迹图已保存：", plot_3d_path)

    if selected_episode_info is not None:
        arm1_final = np.asarray(selected_arm1_traj[-1], dtype=np.float32)
        arm2_final = np.asarray(selected_arm2_traj[-1], dtype=np.float32)
        arm1_dist = float(np.linalg.norm(arm1_final - snapshot["target_1"]))
        arm2_dist = float(np.linalg.norm(arm2_final - snapshot["target_2"]))
        print(f"最终距离: arm1={arm1_dist:.4f}, arm2={arm2_dist:.4f}, success_radius={env.success_distance:.4f}")

    if args.gui and selected_arm1_traj and selected_arm2_traj:
        env.draw_trajectory(selected_arm1_traj, selected_arm2_traj)
        print("GUI 已绘制最后一次测试轨迹，关闭窗口前可持续查看。")
        time.sleep(10)

    env.close()
