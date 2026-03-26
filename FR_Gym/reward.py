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

def cal_success_reward(self, robot_id, target_id, table_id, targettable_id, distance, arm_name="arm"):
    """计算单个机械臂的成功/失败奖励"""
    gripper_joint_indices = [8, 9]

    target_contact_points = p.getContactPoints(bodyA=robot_id, bodyB=target_id)
    table_contact_points = p.getContactPoints(bodyA=robot_id, bodyB=table_id)
    targettable_contact_points = p.getContactPoints(bodyA=robot_id, bodyB=targettable_id)

    gripper_contact = False
    target_contact = False
    table_contact = False
    targettable_contact = False
    other_contact = False

    # 接触目标
    for contact_point in target_contact_points:
        link_index = contact_point[3]

        if link_index in gripper_joint_indices:
            gripper_contact = True

        if link_index not in gripper_joint_indices:
            target_contact = True

    # 接触桌子
    for contact_point in table_contact_points:
        link_index = contact_point[3]
        if link_index not in [0, 1]:
            other_contact = True
            table_contact = True

    # 接触目标台子
    for _ in targettable_contact_points:
        other_contact = True
        targettable_contact = True

    success = judge_success(distance, success_dis=0.02)

    success_reward = 0
    fail = False

    if success and self.step_num <= 100:
        success_reward = 1000
        logger.info("{} 成功抓取！ step={}, distance={}", arm_name, self.step_num, distance)

    elif other_contact:
        success_reward = -100
        fail = True
        if targettable_contact:
            logger.info("{} 失败：碰撞目标台子！ step={}, distance={}", arm_name, self.step_num, distance)
        elif table_contact:
            logger.info("{} 失败：碰撞桌子！ step={}, distance={}", arm_name, self.step_num, distance)

    elif target_contact and not gripper_contact:
        success_reward = -100
        fail = True
        logger.info("{} 失败：非夹爪部位接触目标！ step={}, distance={}", arm_name, self.step_num, distance)

    elif self.step_num > 100:
        success_reward = -100
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
        distance_reward = 0
    else:
        distance_last = getattr(self, last_attr)
        distance_reward = 1000 * (distance_last - distance)

    setattr(self, last_attr, distance)

    return distance_reward



def grasp_reward(self):
    """双机械臂双目标奖励：各抓各的"""
    info = {}

    arm1_reward = 0
    arm2_reward = 0
    distance_reward_1 = 0
    distance_reward_2 = 0
    fail_1 = False
    fail_2 = False

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
        distance_reward_1 = cal_dis_reward(self, distance_1, "arm1")
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
        distance_reward_2 = cal_dis_reward(self, distance_2, "arm2")
        arm2_reward = success_reward_2 + distance_reward_2

        if success_2:
            self.arm2_success = True
            logger.info("arm2 已锁定成功状态")
    else:
        arm2_reward = 0

    total_reward = arm1_reward + arm2_reward

    # 两个都成功才算整个任务成功
    self.success = self.arm1_success and self.arm2_success

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

    return total_reward, info
def judge_success(distance, success_dis=0.02):
    """判断单个机械臂是否抓取成功"""
    return distance < success_dis

def get_distance(self, robot_id, target_id):
    """计算某个机械臂夹爪中心到目标的距离"""
    gripper_link_pos = p.getLinkState(robot_id, 6)[0]
    gripper_rot = R.from_quat(p.getLinkState(robot_id, 7)[1])

    relative_position = np.array([0, 0, 0.15])
    rotated_relative_position = gripper_rot.apply(relative_position)
    gripper_centre_pos = np.array(gripper_link_pos) + rotated_relative_position

    target_position = np.array(p.getBasePositionAndOrientation(target_id)[0])
    distance = np.linalg.norm(gripper_centre_pos - target_position)

    return distance