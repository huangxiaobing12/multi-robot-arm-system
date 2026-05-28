import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import time
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from loguru import logger
import random
from interval import Interval

SINGLE_ARM_SUCCESS_REWARD = 800
JOINT_SUCCESS_REWARD = 4000
JOINT_PROGRESS_REWARD_COEF = 20000
TIMEOUT_FAIL_PENALTY = -500
ROBOT_COLLISION_PENALTY = -800
INCOMPLETE_TASK_PENALTY = -500
LAGGING_DISTANCE_PENALTY_COEF = 20

def cal_success_reward(self, robot_id, target_id, table_id, targettable_id, distance, arm_name="arm"):
    """计算单个机械臂的成功/失败奖励。

    当前任务里用夹爪代理圆柱与目标的接触来判定成功。
    """
    del robot_id, table_id, targettable_id

    proxy_id = self.arm1_proxy if arm_name == "arm1" else self.arm2_proxy
    proxy_contacts = self.p.getContactPoints(bodyA=proxy_id, bodyB=target_id)
    success = len(proxy_contacts) > 0

    if not success:
        success = judge_success(distance, success_dis=self.success_distance)

    success_reward = 0
    fail = False

    if success and self.step_num <= self.max_steps:
        success_reward = SINGLE_ARM_SUCCESS_REWARD
        logger.info("{} 成功接触目标！ step={}, distance={}", arm_name, self.step_num, distance)

    elif self.step_num > self.max_steps:
        success_reward = TIMEOUT_FAIL_PENALTY
        fail = True
        logger.info("{} 失败：执行步数过多！ step={}, distance={}", arm_name, self.step_num, distance)

    return success_reward, success, fail

def cal_dis_reward(self, distance, arm_id):
    """计算某个机械臂的距离奖励"""

    if arm_id == "arm1":
        last_attr = "distance_last_1"
    elif arm_id == "arm2":
        last_attr = "distance_last_2"
    else:
        raise ValueError("arm_id必须是 'arm1' 或 'arm2'")

    if self.step_num == 0:
        distance_delta = 0
        distance_reward = 0
    else:
        distance_last = getattr(self, last_attr)
        if distance_last is None:
            distance_delta = 0
            distance_reward = 0
        else:
            distance_delta = distance_last - distance
            distance_reward = 1000 * distance_delta

    setattr(self, last_attr, distance)

    return distance_reward, distance_delta


def cal_joint_progress_reward(self, distance, arm_id):
    """奖励已成功机械臂之外的另一只机械臂刷新最近目标距离。

    用 episode 内的最近距离而不是单步正向 delta，避免来回晃动时反复刷奖励。
    """
    if arm_id == "arm1":
        best_attr = "joint_progress_best_distance_1"
    elif arm_id == "arm2":
        best_attr = "joint_progress_best_distance_2"
    else:
        raise ValueError("arm_id必须是 'arm1' 或 'arm2'")

    best_distance = getattr(self, best_attr, None)
    if best_distance is None:
        setattr(self, best_attr, distance)
        return 0

    distance_improvement = max(best_distance - distance, 0)
    if distance_improvement > 0:
        setattr(self, best_attr, distance)

    return JOINT_PROGRESS_REWARD_COEF * distance_improvement



def grasp_reward(self):
    """双机械臂双目标奖励：各抓各的"""
    info = {}

    arm1_reward = 0
    arm2_reward = 0
    distance_reward_1 = 0
    distance_reward_2 = 0
    distance_1 = None
    distance_2 = None
    distance_delta_1 = 0
    distance_delta_2 = 0
    fail_1 = False
    fail_2 = False
    robot_collision = check_robot_collision(self)

    if robot_collision:
        distance_1 = get_distance(self, self.fr5, self.target)
        distance_2 = get_distance(self, self.fr5_2, self.target1)
        self.success = False
        self.terminated = True
        self.truncated = False
        self.reward = ROBOT_COLLISION_PENALTY
        info["reward"] = self.reward
        info["is_success"] = False
        info["step_num"] = self.step_num
        info["arm1_success"] = self.arm1_success
        info["arm2_success"] = self.arm2_success
        info["success_reward"] = 0
        info["distance_reward"] = 0
        info["joint_success_reward"] = 0
        info["joint_progress_reward"] = 0
        info["lagging_distance_penalty"] = 0
        info["incomplete_task_penalty"] = 0
        info["arm1_distance"] = distance_1
        info["arm2_distance"] = distance_2
        info["robot_collision"] = True
        info["robot_collision_details"] = getattr(self, "last_robot_collision_details", "")
        logger.info("双机械臂发生碰撞！ step={} {}", self.step_num, info["robot_collision_details"])
        return self.reward, info

    # ---------- arm1 ----------
    if not self.arm1_success:
        distance_1 = get_distance(self, self.fr5, self.target)
        success_reward_1, success_1, fail_1 = cal_success_reward(
            self,
            robot_id=self.fr5,
            target_id=self.target,
            table_id=self.table,
            targettable_id=self.targettable,
            distance=distance_1,
            arm_name="arm1"
        )
        distance_reward_1, distance_delta_1 = cal_dis_reward(self, distance_1, "arm1")
        arm1_reward = success_reward_1 + distance_reward_1

        # 一旦成功，永久记录
        if success_1:
            self.arm1_success = True
            logger.info("arm1 已锁定成功状态")
    else:
        # 已经成功后，不再计算奖励/失败
        arm1_reward = 0

    # ---------- arm2 ----------
    if not self.arm2_success:
        distance_2 = get_distance(self, self.fr5_2, self.target1)
        success_reward_2, success_2, fail_2 = cal_success_reward(
            self,
            robot_id=self.fr5_2,
            target_id=self.target1,
            table_id=self.table,
            targettable_id=self.targettable1,
            distance=distance_2,
            arm_name="arm2"
        )
        distance_reward_2, distance_delta_2 = cal_dis_reward(self, distance_2, "arm2")
        arm2_reward = success_reward_2 + distance_reward_2

        if success_2:
            self.arm2_success = True
            logger.info("arm2 已锁定成功状态")
    else:
        arm2_reward = 0

    # 两个都成功才算整个任务成功
    self.success = self.arm1_success and self.arm2_success
    joint_success_reward = JOINT_SUCCESS_REWARD if self.success and not (fail_1 or fail_2) else 0
    joint_progress_reward = 0
    lagging_distance_penalty = 0
    if self.arm1_success and not self.arm2_success:
        joint_progress_reward = cal_joint_progress_reward(self, distance_2, "arm2")
        lagging_distance_penalty = -LAGGING_DISTANCE_PENALTY_COEF * distance_2
    elif self.arm2_success and not self.arm1_success:
        joint_progress_reward = cal_joint_progress_reward(self, distance_1, "arm1")
        lagging_distance_penalty = -LAGGING_DISTANCE_PENALTY_COEF * distance_1

    incomplete_task_penalty = INCOMPLETE_TASK_PENALTY if (fail_1 or fail_2) and not self.success else 0

    total_reward = (
        arm1_reward
        + arm2_reward
        + joint_success_reward
        + joint_progress_reward
        + lagging_distance_penalty
        + incomplete_task_penalty
    )

    # 任意一个未完成机械臂失败，则终止；或者两个都成功，也终止
    if fail_1 or fail_2 or self.success:
        self.terminated = True
    else:
        self.terminated = False

    self.truncated = False
    self.reward = total_reward

    info["reward"] = total_reward
    info["is_success"] = self.success
    info["step_num"] = self.step_num
    info["arm1_success"] = self.arm1_success
    info["arm2_success"] = self.arm2_success
    info["success_reward"] = int(self.success)
    info["distance_reward"] = distance_reward_1 + distance_reward_2
    info["joint_success_reward"] = joint_success_reward
    info["joint_progress_reward"] = joint_progress_reward
    info["lagging_distance_penalty"] = lagging_distance_penalty
    info["incomplete_task_penalty"] = incomplete_task_penalty
    info["arm1_distance"] = distance_1 if distance_1 is not None else 0
    info["arm2_distance"] = distance_2 if distance_2 is not None else 0
    info["robot_collision"] = False
    info["robot_collision_details"] = ""

    return total_reward, info
def judge_success(distance, success_dis=0.02):
    """判断单个机械臂是否抓取成功"""
    return distance < success_dis

def get_distance(self, robot_id, target_id):
    """计算某个机械臂夹爪中心到目标的距离"""
    gripper_link_pos = self.p.getLinkState(robot_id, 6)[0]
    gripper_rot = R.from_quat(self.p.getLinkState(robot_id, 7)[1])

    relative_position = np.array([0, 0, 0.15])
    rotated_relative_position = gripper_rot.apply(relative_position)
    gripper_centre_pos = np.array(gripper_link_pos) + rotated_relative_position

    target_position = np.array(self.p.getBasePositionAndOrientation(target_id)[0])
    distance = np.linalg.norm(gripper_centre_pos - target_position)

    return distance


def check_robot_collision(self):
    """检测两台机械臂之间是否发生碰撞。"""
    contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.fr5_2)
    active_contacts = [
        contact_point
        for contact_point in contact_points
        if contact_point[8] <= 0 or contact_point[9] > 1e-6
    ]
    self.last_robot_collision_details = describe_robot_collision(self, active_contacts)
    return len(active_contacts) > 0


def get_link_name(self, robot_id, link_index):
    """返回 PyBullet link 名称，-1 表示 base。"""
    if link_index == -1:
        return "base"
    link_name = self.p.getJointInfo(robot_id, link_index)[12]
    if isinstance(link_name, bytes):
        return link_name.decode("utf-8", errors="ignore")
    return str(link_name)


def describe_robot_collision(self, contact_points, max_items=3):
    """生成双臂碰撞的简短 link 级日志。"""
    if not contact_points:
        return ""

    details = []
    for contact_point in contact_points[:max_items]:
        link_a = contact_point[3]
        link_b = contact_point[4]
        distance = contact_point[8]
        normal_force = contact_point[9]
        details.append(
            "arm1:{}({}) <-> arm2:{}({}), distance={:.5f}, force={:.3f}".format(
                get_link_name(self, self.fr5, link_a),
                link_a,
                get_link_name(self, self.fr5_2, link_b),
                link_b,
                distance,
                normal_force,
            )
        )

    return "; ".join(details)
