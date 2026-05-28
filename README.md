<!--
'''
 @Author: Zhanxin Geng
 @Date: 2026-05-28 
 @Last Modified by:   Zhanxin Geng
 @Last Modified time: 2026-05-28 15:04:04 
'''
-->
# FR_Reinforcement learning
中文版跳转   [中文版Readme](README_cn.md)
## Introduction
This project is a reinforcement learning training code for grasping with the FAIRINO FR5 robotic arm, based on pybullet and stable baseline3.

For the video, please refer to Bilibili (please like, share, and subscribe!) 

## I. Hardware Overview
Training hardware: NVIDIA GeForce 3090 graphics processor and Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz.
Deployment hardware and software: NVIDIA GeForce RTX 3070 graphics processor and 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz, using Ubuntu 20.04 with ROS Noetic.

## II.Scene Deployment Instructions
The FAIRINO FR5 is a high-precision industrial six-axis robotic arm with a repeatability of 0.02mm. We have added a two-finger gripper PGI-140-80 from DH Robotics to the end of the arm, which can achieve an effective stroke of 80mm and a maximum gripping force of 140N. In this experiment, we use pybullet as the simulation platform.

## III. Requirments
gym==0.26.2

pybullet

opencv-python

loguru

stable_baselines3

scipy

numpy

## IV. Code Explanation
- fr5_description: Stores the URDF model files of the robotic arm.
- Fr5_env.py: Constructs the reinforcement learning environment.
- Fr5_train.py: Contains the reinforcement learning training code.
- FR5_test.py: Used for reinforcement learning testing.
## 五、How to use
Available algorithms：
- PPO（default）
- A2C
- DDPG
- TD3

Start training:
```python
python Fr5_train.py --timesteps 30000 --gui False
```
Visualize training results with TensorBoard:

```
tensorboard --logdir .../logs/PPO/your_training_results --port 6006
```
Infer the model:
```
python Fr5_test.py --model_path your_model_dir --gui True
```

## Citation
If you use this project in your research, please cite it:

```bibtex
```bibtex
@misc{ZhanxinGeng,
  title        = {multi-robot-arm-system},
  author       = {ZhanxinGeng、 XiaobingHuang},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/huangxiaobing12/multi-robot-arm-system}
}
```


