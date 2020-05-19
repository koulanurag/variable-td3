import gym

from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper


class MujocoConfig(BaseConfig):
    def __init__(self):
        super(MujocoConfig, self).__init__(max_env_steps=int(1e6))

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, eval=False):
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return MultiStepWrapper(env)


run_config = MujocoConfig()
