# FR_Reinforcement learning

## 介绍
本项目是基于pybullet和stable baseline3 的法奥机械臂的强化学习抓取训练代码



## 一、安装机器简介
训练的硬件：NVIDIA GeForce 3090图形处理器和Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
部署硬件及软件：NVIDIA GeForce RTX 3070图形处理器和11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz，使用系统为Ubuntu 20.04+ROS Noetic

## 二、场景部署说明
FAIRINO FR5 是一个高精度的工业六轴机械臂，它的重复定位进度达到了0.02mm，我们在桌子上大件路两个机械臂，同事跟踪两个目标，过程中防止碰撞，目前的场景为左边机械臂需要到达右边目标，右边机械臂到达左边目标，交叉到达目标点，过程中不允许发生碰撞。

## 三、requirments 必须的安装包
gym==0.26.2

pybullet

opencv-python

loguru

stable_baselines3

scipy

numpy

## 四、代码说明
fr5_description
用于储存机械臂urdf模型文件
Fr5_env.py
用于构建强化学习环境
Fr5_train.py
强化学习训练代码
FR5_test.py
用于强化学习测试
## 五、How to use
可使用的算法：
- PPO（默认）
- A2C
- DDPG
- TD3

 开始训练：
```python
python Fr5_train.py --timesteps 30000 --gui False
```
tensorboard可视化训练结果

```
tensorboard --logdir .../logs/PPO/你的训练结果 --port 6006
```
推理模型
```
python Fr5_test.py --model_path your_model_dir --gui True
```

## 引用
如果你的研究/项目使用了本仓库的代码，请引用：

```bibtex
@misc{ZhanxinGeng,
  title        = {multi-robot-arm-system},
  author       = {ZhanxinGeng、 XiaobingHuang},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/huangxiaobing12/multi-robot-arm-system}
}
```
