import gym

from core.config import BaseConfig


class Box2DConfig(BaseConfig):
    def __init__(self):
        super(Box2DConfig, self).__init__(max_env_steps=int(5e5),
                                          start_step=int(1e4),
                                          lr=1e-3,
                                          replay_memory_capacity=int(1e5),
                                          fixed_action_repeat=1,
                                          test_interval_steps=5000)

    def new_game(self, seed=None):
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)
        return env


run_config = Box2DConfig()
