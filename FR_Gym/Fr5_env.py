'''
 @Author: Zhanxin Geng
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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
from .reward import grasp_reward
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
URDF_PATH = os.path.join(ROOT, "fr5_description", "urdf", "fr5v6.urdf")
TABLE_PATH = os.path.join(ROOT, "env", "table", "table.urdf")
class FR5_Env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,gui = False):
        super(FR5_Env).__init__()
        self.gui = gui
        self.success_distance = 0.02
        self.max_steps = 225
        self.gripper_proxy_radius = 0.012
        self.gripper_proxy_height = 0.04
        self.fixed_target_1 = [0.0, 0.6, 0.22]
        self.fixed_target_2 = [0.0, 0.4, 0.22]
        self.step_num = 0
        self.Con_cube = None
        # self.last_success = False

        # 设置最小的关节变化量
        low_action = np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
        high_action = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32,
        )

        # 初始化pybullet环境
        if gui == False:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        else :
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
        # self.p.setTimeStep(1/240)
        # print(self.p)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.arm1_success = False
        self.arm2_success = False
        self._camera_configured = False
        # 初始化环境
        self.init_env()

    def init_env(self):
        '''
            仿真环境初始化
        '''
        # boxId = self.p.loadURDF("plane.urdf")
        # 创建机械臂
        self.fr5 = self.p.loadURDF(
            URDF_PATH,
            useFixedBase=True,
            basePosition=[0, 0.05, 0],
            baseOrientation=self.p.getQuaternionFromEuler([0, 0, np.pi]),
            flags=self.p.URDF_USE_SELF_COLLISION,
        )
        self.arm1_success = False
        self.arm2_success = False
        # self.fr5 = self.p.loadURDF("FR5_Reinforcement-learning/fr5_description/urdf/fr5v6.urdf",useFixedBase=True, basePosition=[0, 0, 0],
        #                       baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),flags = p.URDF_USE_SELF_COLLISION)

        self.fr5_2 = self.p.loadURDF(
            URDF_PATH,
            useFixedBase=True,
            basePosition=[0, 0.95, 0],
            baseOrientation=self.p.getQuaternionFromEuler([0, 0, 0]),
            flags=self.p.URDF_USE_SELF_COLLISION
        )
        # 创建桌子
        self.table = self.p.loadURDF(
            TABLE_PATH,
            basePosition=[0, 0.5, -0.63],
            baseOrientation=self.p.getQuaternionFromEuler([0, 0, np.pi / 2]),
        )

        # 创建目标
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                          radius=0.02,height = 0.05)
        self.target = self.p.createMultiBody(baseMass=0,  # 质量
                           baseCollisionShapeIndex=collisionTargetId,
                           basePosition=self.fixed_target_1) 
        self.target1 = self.p.createMultiBody(baseMass=0,  # 质量
                           baseCollisionShapeIndex=collisionTargetId,
                           basePosition=self.fixed_target_2) 
        
        # 创建目标杯子的台子
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                            radius=0.03,height = 0.3)
        self.targettable = self.p.createMultiBody(baseMass=0,  # 质量
                            baseCollisionShapeIndex=collisionTargetId,
                            basePosition=[self.fixed_target_1[0], self.fixed_target_1[1], self.fixed_target_1[2] - 0.175]) 
        self.targettable1 = self.p.createMultiBody(baseMass=0,  # 质量
                            baseCollisionShapeIndex=collisionTargetId,
                            basePosition=[self.fixed_target_2[0], self.fixed_target_2[1], self.fixed_target_2[2] - 0.175])

        self.arm1_proxy = self.create_gripper_proxy([0.95, 0.2, 0.2, 0.7])
        self.arm2_proxy = self.create_gripper_proxy([0.2, 0.45, 0.95, 0.7])
        self.disable_proxy_self_collision(self.arm1_proxy, self.fr5)
        self.disable_proxy_self_collision(self.arm2_proxy, self.fr5_2)
        self.update_gripper_proxies()

    def step(self, action):
        """step"""
        info = {}

        joint_ids = [1, 2, 3, 4, 5, 6]
        ctrl_ids = [1, 2, 3, 4, 5, 6, 8, 9]

        action = np.array(action, dtype=np.float32)

        if action.shape[0] != 12:
            raise ValueError(f"action维度应该是12, 但当前是 {action.shape[0]}")

        action_1 = action[0:6]
        action_2 = action[6:12]

        # ---------- 机械臂1 ----------
        joint_angles_1 = []
        for i in joint_ids:
            joint_info = self.p.getJointState(self.fr5, i)
            joint_angles_1.append(joint_info[0])

        if not self.arm1_success:
            target_joint_angles_1 = np.array(joint_angles_1) + (action_1 / 180.0 * np.pi)
        else:
            # 已成功，保持当前位置不动
            target_joint_angles_1 = np.array(joint_angles_1)

        gripper_1 = np.array([0.0, 0.0])
        target_positions_1 = np.hstack([target_joint_angles_1, gripper_1])

        self.p.setJointMotorControlArray(
            self.fr5,
            ctrl_ids,
            self.p.POSITION_CONTROL,
            targetPositions=target_positions_1
        )

        # ---------- 机械臂2 ----------
        joint_angles_2 = []
        for i in joint_ids:
            joint_info = self.p.getJointState(self.fr5_2, i)
            joint_angles_2.append(joint_info[0])

        if not self.arm2_success:
            target_joint_angles_2 = np.array(joint_angles_2) + (action_2 / 180.0 * np.pi)
        else:
            target_joint_angles_2 = np.array(joint_angles_2)

        gripper_2 = np.array([0.0, 0.0])
        target_positions_2 = np.hstack([target_joint_angles_2, gripper_2])

        self.p.setJointMotorControlArray(
            self.fr5_2,
            ctrl_ids,
            self.p.POSITION_CONTROL,
            targetPositions=target_positions_2
        )

        # 推进一步仿真
        for _ in range(20):
            self.p.stepSimulation()

        self.update_gripper_proxies()

        # 计算奖励
        self.reward, info = grasp_reward(self)

        # 更新观测
        self.get_observation()

        self.step_num += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        '''重置环境参数'''
        self.arm1_success = False
        self.arm2_success = False
        self.step_num = 0
        self.reward = 0
        self.terminated = False
        self.success = False
        joint_ids = [1,2,3,4,5,6,8,9]
        # 重新设置机械臂的位置。左右底座已做 180 度中心对称，关节角保持同一套局部姿态。
        neutral_angle_1 = [-49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118, -49.45849125928217, 0, 0, 0]
        neutral_angle_2 = [-49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118, -49.45849125928217, 0, 0, 0]
        neutral_angle_1 = [x * math.pi / 180 for x in neutral_angle_1]
        neutral_angle_2 = [x * math.pi / 180 for x in neutral_angle_2]
        self.p.setJointMotorControlArray(
            self.fr5,
            [1, 2, 3, 4, 5, 6, 8, 9],
            self.p.POSITION_CONTROL,
            targetPositions=neutral_angle_1,
        )
        self.p.setJointMotorControlArray(
            self.fr5_2,
            joint_ids,
            self.p.POSITION_CONTROL,
            targetPositions=neutral_angle_2
        )
        # # 重新设置目标位置
        # self.goalx = np.random.uniform(-0.2, 0.2, 1)
        # self.goaly = np.random.uniform(0.6, 0.8, 1)
        # self.goalz = np.random.uniform(0.1, 0.3, 1)
        self.goalx, self.goaly, self.goalz = self.fixed_target_1
        self.goalx1, self.goaly1, self.goalz1 = self.fixed_target_2
        self.target_position = [self.goalx, self.goaly, self.goalz]
        self.targettable_position = [self.goalx, self.goaly, self.goalz-0.175]
        self.target_position1 = [self.goalx1, self.goaly1, self.goalz1]
        self.targettable_position1 = [self.goalx1, self.goaly1, self.goalz1-0.175]
        self.p.resetBasePositionAndOrientation(self.targettable,self.targettable_position, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.target,self.target_position, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.targettable1,self.targettable_position1, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.target1,self.target_position1, [0, 0, 0, 1])
        self.distance_last_1 = None
        self.distance_last_2 = None
        
        
        for i in range(100):
            self.p.stepSimulation()
            # time.sleep(10./240.)

        self.update_gripper_proxies()
        self.get_observation()
        
        
        infos = {}
        infos['is_success'] = False
        infos['reward'] = 0
        infos['step_num'] = 0
        return self.observation,infos

    def get_single_arm_observation(self, robot_id, target_id, add_noise=False):
        """单个机械臂对应单个目标的观测"""

        # 末端位置
        gripper_pos = self.p.getLinkState(robot_id, 6)[0]
        gripper_pos = np.array(gripper_pos)

        relative_position = np.array([0, 0, 0.15])

        # 夹爪中心位置
        rotation = R.from_quat(self.p.getLinkState(robot_id, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = gripper_pos + rotated_relative_position

        # 6个关节角
        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = self.p.getJointState(robot_id, i)
            joint_angles[i - 1] = joint_info[0] * 180 / np.pi
            if add_noise:
                joint_angles[i - 1] = self.add_noise(joint_angles[i - 1], range=0, gaussian=True)

        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2

        obs_gripper_centre_pos = np.array([
            (gripper_centre_pos[0] + 0.922) / 1.844,
            (gripper_centre_pos[1] + 0.922) / 1.844,
            (gripper_centre_pos[2] + 0.5) / 1
        ], dtype=np.float32)

        # 对应目标位置
        target_position = np.array(self.p.getBasePositionAndOrientation(target_id)[0], dtype=np.float32)

        obs_target_position = np.array([
            (target_position[0] + 0.2) / 0.4,
            target_position[1] / 1.0,
            (target_position[2] - 0.1) / 0.2
        ], dtype=np.float32)

        obs = np.hstack((
            obs_gripper_centre_pos,   # 3
            obs_joint_angles,         # 6
            obs_target_position       # 3
        )).astype(np.float32)

        return obs
    def get_observation(self, add_noise=False):
        """两个机械臂、两个不同目标的 observation"""

        # 机械臂1 -> target
        obs_arm1 = self.get_single_arm_observation(self.fr5, self.target, add_noise)

        # 机械臂2 -> target1
        obs_arm2 = self.get_single_arm_observation(self.fr5_2, self.target1, add_noise)

        # 拼接成总观测
        self.observation = np.hstack((obs_arm1, obs_arm2)).astype(np.float32)

    def get_gripper_center(self, robot_id):
        """返回夹爪中心的世界坐标。"""
        gripper_pos, gripper_quat = self.get_gripper_pose(robot_id)
        rotation = R.from_quat(gripper_quat)
        relative_position = np.array([0, 0, 0.15], dtype=np.float32)
        return gripper_pos + rotation.apply(relative_position)

    def get_gripper_pose(self, robot_id):
        """返回末端夹爪参考姿态。"""
        gripper_pos = np.array(self.p.getLinkState(robot_id, 6)[0], dtype=np.float32)
        gripper_quat = np.array(self.p.getLinkState(robot_id, 7)[1], dtype=np.float32)
        return gripper_pos, gripper_quat

    def create_gripper_proxy(self, rgba_color):
        """创建一个随夹爪移动的小圆柱代理体。"""
        collision_id = self.p.createCollisionShape(
            shapeType=self.p.GEOM_CYLINDER,
            radius=self.gripper_proxy_radius,
            height=self.gripper_proxy_height,
        )
        visual_id = self.p.createVisualShape(
            shapeType=self.p.GEOM_CYLINDER,
            radius=self.gripper_proxy_radius,
            length=self.gripper_proxy_height,
            rgbaColor=rgba_color,
        )
        return self.p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
        )

    def disable_proxy_self_collision(self, proxy_id, robot_id):
        """关闭代理体和所属机械臂自身的碰撞，保留与外界的接触。"""
        for link_index in range(-1, self.p.getNumJoints(robot_id)):
            self.p.setCollisionFilterPair(proxy_id, robot_id, -1, link_index, 0)

    def update_gripper_proxies(self):
        """同步两个夹爪代理体到当前夹爪中心位置。"""
        arm1_center = self.get_gripper_center(self.fr5)
        _, arm1_quat = self.get_gripper_pose(self.fr5)
        self.p.resetBasePositionAndOrientation(self.arm1_proxy, arm1_center, arm1_quat)

        arm2_center = self.get_gripper_center(self.fr5_2)
        _, arm2_quat = self.get_gripper_pose(self.fr5_2)
        self.p.resetBasePositionAndOrientation(self.arm2_proxy, arm2_center, arm2_quat)

    def get_debug_snapshot(self):
        """返回调试/可视化所需的关键位置。"""
        return {
            "arm1_gripper_center": self.get_gripper_center(self.fr5),
            "arm2_gripper_center": self.get_gripper_center(self.fr5_2),
            "target_1": np.array(self.p.getBasePositionAndOrientation(self.target)[0], dtype=np.float32),
            "target_2": np.array(self.p.getBasePositionAndOrientation(self.target1)[0], dtype=np.float32),
        }

    def draw_trajectory(self, arm1_points, arm2_points):
        """在 GUI 中绘制两只机械臂的末端轨迹。"""
        if not self.gui:
            return

        self.p.removeAllUserDebugItems()
        self.render()

        snapshot = self.get_debug_snapshot()
        self.p.addUserDebugText("target1", snapshot["target_1"], [1, 0.2, 0.2], textSize=1.2)
        self.p.addUserDebugText("target2", snapshot["target_2"], [0.2, 0.6, 1], textSize=1.2)

        for start, end in zip(arm1_points[:-1], arm1_points[1:]):
            self.p.addUserDebugLine(start, end, [1, 0, 0], lineWidth=2.0, lifeTime=0)

        for start, end in zip(arm2_points[:-1], arm2_points[1:]):
            self.p.addUserDebugLine(start, end, [0, 0.4, 1], lineWidth=2.0, lifeTime=0)

        # return self.observation
    # def get_observation(self,add_noise = False):
    #     """计算observation"""
    #     Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
    #     Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
    #     Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
    #     relative_position = np.array([0, 0, 0.15])
        
    #     # 固定夹爪相对于机械臂末端的相对位置转换
    #     rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
    #     rotated_relative_position = rotation.apply(relative_position)
    #     # print([Gripper_posx, Gripper_posy,Gripper_posz])
    #     gripper_centre_pos = [Gripper_posx, Gripper_posy,Gripper_posz] + rotated_relative_position

    #     joint_angles = [0,0,0,0,0,0]
    #     for i in [1,2,3,4,5,6]:
    #         joint_info = p.getJointState(self.fr5, i)
    #         joint_angles[i-1]  = joint_info[0]*180/np.pi  # 第一个元素是当前关节角度
    #         if add_noise == True:
    #             joint_angles[i-1] = self.add_noise(joint_angles[i-1],range=0,gaussian=True)
    #     # print("joint_angles",str(joint_angles))
    #     # print("gripper_centre_pos",str(gripper_centre_pos))

    #     # 计算夹爪的朝向
    #     gripper_orientation = p.getLinkState(self.fr5, 7)[1]
    #     gripper_orientation = R.from_quat(gripper_orientation)
    #     gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)

    #     # 计算obs
    #     obs_joint_angles = ((np.array(joint_angles,dtype=np.float32)/180)+1)/2
        
    #     # gripper_centre_pos[0] = self.add_noise(gripper_centre_pos[0],range=0.005,gaussian=True)
    #     # gripper_centre_pos[1] = self.add_noise(gripper_centre_pos[1],range=0.005,gaussian=True)
    #     # gripper_centre_pos[2] = self.add_noise(gripper_centre_pos[2],range=0.005,gaussian=True)
    #     obs_gripper_centre_pos = np.array([(gripper_centre_pos[0]+0.922)/1.844,
    #                                        (gripper_centre_pos[1]+0.922)/1.844,
    #                                        (gripper_centre_pos[2]+0.5)/1],dtype=np.float32)
        
    #     obs_gripper_orientation = (np.array([gripper_orientation[0],gripper_orientation[1],gripper_orientation[2]],dtype=np.float32)+180)/360
        
    #     self.target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])

    #     obs_target_position = np.array([(self.target_position[0]+0.2)/0.4,
    #                                     (self.target_position[1]-0.6)/0.2,
    #                                     (self.target_position[2]-0.1)/0.2],dtype=np.float32)

    #     self.observation = np.hstack((obs_gripper_centre_pos,obs_joint_angles,obs_target_position),dtype=np.float32).flatten()

    #     self.observation = self.observation.flatten()
    #     self.observation = self.observation.reshape(1,12)
        # self.observation = np.hstack((np.array(joint_angles,dtype=np.float32),target_delta_position[0]),dtype=np.float32)


    def render(self):
        '''设置观察角度'''
        if self.gui and not self._camera_configured:
            self.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=90,
                cameraPitch=-7.6,
                cameraTargetPosition=[0.39, 0.45, 0.42],
            )
            self._camera_configured = True
    
    def close(self):
        self.p.disconnect()

    def add_noise(self, angle, range, gaussian=False):
        '''添加噪声'''
        if gaussian:
            angle += np.clip(np.random.normal(0, 1) * range, -1, 1)
        else:
            angle += random.uniform(-5, 5)
        return angle

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env 
    Env = FR5_Env(gui=True)
    Env.reset()
    check_env(Env, warn=True)
    # for i in range(100):
    #         p.stepSimulation()
    #         time.sleep(1./240.)
    Env.render()
    print("test going")
    time.sleep(10)
    # observation, reward, terminated, truncated, info = Env.step([0,0,0,0,0,20])
    # print(reward)
    time.sleep(100)
