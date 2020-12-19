import gym
from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper
import numpy as np


class CassieConfigV2(BaseConfig):
    def __init__(self):
        super(CassieConfigV2, self).__init__(max_env_steps=int(1e6),
                                             start_step=int(25e3),
                                             lr=3e-4,
                                             replay_memory_capacity=int(1e6),
                                             fixed_action_repeat=1,
                                             test_interval_steps=5000)

    def new_game(self, seed=None, save_video=False, video_dir_path=None, uid=None):
        from .cassie import CassieEnv_v2
        env = CassieEnv_v2()
        env.reward_range = None
        env.close = lambda: None
        from gym import spaces
        env.action_space = Box(low=-3 * np.ones(env.action_space.shape),
                               high=3 * np.ones(env.action_space.shape),
                               dtype=np.float32)
        return MultiStepWrapper(env)


run_config = CassieConfigV2()
