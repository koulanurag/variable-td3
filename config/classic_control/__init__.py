import gym

from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper


class ClassicControlConfig(BaseConfig):
    def __init__(self):
        super(ClassicControlConfig, self).__init__(max_env_steps=int(2e5),
                                                   start_step=int(1e4),
                                                   lr=1e-3,
                                                   replay_memory_capacity=int(2e5),
                                                   fixed_action_repeat=1,
                                                   test_interval_steps=2000)

    def new_game(self, seed=None):
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)
            
        return MultiStepWrapper(env)


run_config = ClassicControlConfig()
