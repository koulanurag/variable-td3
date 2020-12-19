import gym
from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper
import numpy as np


class CassieConfig(BaseConfig):
    def __init__(self):
        super(CassieConfig, self).__init__(max_env_steps=int(1e6),
                                           start_step=int(25e3),
                                           lr=3e-4,
                                           replay_memory_capacity=int(1e6),
                                           fixed_action_repeat=1,
                                           test_interval_steps=5000)

    def new_game(self, seed=None, save_video=False, video_dir_path=None, uid=None):
        import gym_cassie
        env = gym.make(self.env_name)
        env.reward_range = None
        env.close = lambda: None
        from gym import spaces
        env.action_space = spaces.Box(low=-3 * np.ones(env.action_space.low.shape),
                                      high=3 * np.ones(env.action_space.high.shape),
                                      dtype=np.float32)
        return MultiStepWrapper(env)


run_config = CassieConfig()
