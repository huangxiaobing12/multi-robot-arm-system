'''
 @Author: Prince Wang 
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''


import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# sys.path.append(r"FR5_Reinforcement-learning\utils")

from stable_baselines3 import A2C,PPO,DDPG,TD3,SAC
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from .Fr5_env import FR5_Env
import time

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback,CallbackList,BaseCallback,CheckpointCallback
from .Callback import TensorboardCallback, ViewerCallback
from loguru import logger
from utils.arguments import get_args

now = time.strftime('%m%d-%H%M%S', time.localtime())
args, kwargs = get_args()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models_dir = args.models_dir
logs_dir = args.logs_dir
checkpoints = args.checkpoints
test = args.test

def make_env(i, gui=False, monitor_dir=None):
    def _init():
        env = FR5_Env(gui=gui)
        if monitor_dir is not None:
            env = Monitor(env, monitor_dir)
        if gui:
            env.render()
        env.reset()
        return env
    set_random_seed(i)
    return _init

if __name__ == '__main__':
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):    
        os.makedirs(logs_dir)
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    if not os.path.exists(test):
        os.makedirs(test)
    import pybullet_data
    print(pybullet_data.getDataPath())
    # Instantiate the env
    num_train = 16
    if num_train == 1:
        env = DummyVecEnv([make_env(0, gui=args.gui, monitor_dir=logs_dir)])
    else:
        env = SubprocVecEnv([
            make_env(i, gui=False, monitor_dir=logs_dir)
            for i in range(num_train)
        ])
    eval_env = DummyVecEnv([make_env(10_000, gui=False, monitor_dir=test)])
    viewer_env = None
    
    new_logger = configure(logs_dir, ["stdout", "csv", "tensorboard"])

    # HACK
    # Define and Train the agent
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir,batch_size=256,device="cuda")
    best_model_path = os.path.join(models_dir, "best_model.zip")
    resume_model_path = args.model_path if os.path.isfile(args.model_path) else best_model_path
    if os.path.isfile(resume_model_path):
        model = PPO.load(resume_model_path, env=env, device="cuda")
        print("✅ Loaded:", resume_model_path)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir, batch_size=256, device="cuda")
        print("🆕 New model created")
    # model = SAC("MlpPolicy",env, verbose=1, tensorboard_log=logs_dir,batch_size=256,device="cuda",gamma = 0.9,learning_rate = 0.00001)
    # model = PPO(policy = "MlpPolicy",
    #         env = env,
    #         learning_rate = 0.0003,
    #         n_steps = 2048,
    #         batch_size = 256,
    #         n_epochs = 10,
    #         gamma = 0.99,
    #         gae_lambda = 0.95,
    #         clip_range=  0.2,
    #         clip_range_vf = None,
    #         normalize_advantage = True,
    #         ent_coef = 0,
    #         vf_coef = 0.5,
    #         max_grad_norm = 0.5,
    #         use_sde = True,
    #         sde_sample_freq = -1,
    #         target_kl = None,
    #         stats_window_size = 100,
    #         tensorboard_log = logs_dir,
    #         policy_kwargs = dict(normalize_images=False),
    #         verbose = 1,
    #         seed = None,
    #         device = "cuda",
    #         _init_setup_model = True)

    model.set_logger(new_logger)
    tensorboard_callback = TensorboardCallback()
    
    # 创建测试环境回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=test,
        eval_freq=3000,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )

    # 多环境训练时，额外在主进程开一个常驻 GUI 环境做观察/调试
    if args.gui and num_train > 1:
        viewer_env = DummyVecEnv([make_env(20_000, gui=True, monitor_dir=None)])
        viewer_callback = ViewerCallback(viewer_env=viewer_env, step_freq=20)

    TIMESTEPS = args.timesteps
    for eposide in range(1000):
        # 创建 CheckpointCallback 实例来保存模型检查点
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoints)
        callbacks_with_checkpoint = [eval_callback, checkpoint_callback, tensorboard_callback]
        if args.gui and num_train > 1:
            callbacks_with_checkpoint.append(viewer_callback)
        model.learn(total_timesteps=TIMESTEPS,
                    tb_log_name=f"PPO-run-eposide{eposide}", # TensorBoard 日志运行的名称
                    reset_num_timesteps=False,  # 是否重置模型的当前时间步数
                    callback=CallbackList(callbacks_with_checkpoint),  # 在每一步调用的回调，可以用CheckpointCallback来创建一个存档点和规定存档间隔。
                    log_interval=10  #  记录一次信息的时间步数
                    )
        
        # 保存模型
        if eposide % 5 == 0:
            model.save(models_dir+f"/PPO-run-eposide{eposide}")
            logger.info(f"**************eposide--{eposide} saved**************")

    env.close()
    eval_env.close()
    if viewer_env is not None:
        viewer_env.close()
