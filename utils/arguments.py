"""
Author: wangziyuan 13536655301
Date: 2024-04-10 22:55:27
LastEditors: wangziyuan 13536655301
LastEditTime: 2024-05-09 16:16:12
"""

import argparse
import os
import time

# 当前脚本所在目录（绝对路径）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 时间戳
now = time.strftime('%m%d-%H%M%S', time.localtime())


def get_args():
    parser = argparse.ArgumentParser(description="Running time configurations")

    # ===== 路径类参数（全部改为绝对路径 + os.path.join） =====
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.path.join(BASE_DIR, "models", "PPO", "best_model.zip")
    )

    parser.add_argument(
        '--models_dir',
        type=str,
        default=os.path.join(BASE_DIR, "models", "PPO")
    )

    parser.add_argument(
        '--logs_dir',
        type=str,
        default=os.path.join(BASE_DIR, "logs", "PPO", now)
    )

    parser.add_argument(
        '--checkpoints',
        type=str,
        default=os.path.join(BASE_DIR, "checkpoints", "PPO", now)
    )

    parser.add_argument(
        '--test',
        type=str,
        default=os.path.join(BASE_DIR, "logs", "test", now)
    )

    # ===== 其它参数 =====
    parser.add_argument('--test_num', type=int, default=100)

    # ⚠ bool 参数不能用 type=bool（会出错）
    # 改成 flag 形式
    parser.add_argument('--gui', action='store_true', help="Enable GUI")

    parser.add_argument('--timesteps', type=int, default=2000)#30000

    args = parser.parse_args()
    kwargs = vars(args)

    return args, kwargs