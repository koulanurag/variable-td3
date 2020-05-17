from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper
import gym


class Box2DConfig(BaseConfig):
    def __init__(self):
        super(Box2DConfig, self).__init__(max_env_steps=int(10e5),
                                          lr=1e-3,
                                          tau=0.01,
                                          replay_memory_capacity=100000,
                                          fixed_action_repeat=1)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, eval=False):
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return MultiStepWrapper(env)


run_config = Box2DConfig()
