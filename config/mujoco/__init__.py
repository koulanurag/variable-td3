import gym

from core.config import BaseConfig


class MujocoConfig(BaseConfig):
    def __init__(self):
        super(MujocoConfig, self).__init__(max_env_steps=int(1e6),
                                           start_step=int(25e3),
                                           lr=3e-4,
                                           replay_memory_capacity=int(1e6),
                                           fixed_action_repeat=1,
                                           test_interval_steps=5000)

    def new_game(self, seed=None):
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)
        return env


run_config = MujocoConfig()
