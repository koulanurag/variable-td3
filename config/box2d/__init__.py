import gym

from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper


class Box2DConfig(BaseConfig):
    def __init__(self):
        super(Box2DConfig, self).__init__(max_env_steps=int(2e5),
                                          lr=1e-3,
                                          replay_memory_capacity=50000,
                                          fixed_action_repeat=1)

    def new_game(self, seed=None, save_video=False, video_dir_path=None, uid=None):
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=video_dir_path, force=True, video_callable=lambda episode_id: True, uid=uid)
        return MultiStepWrapper(env)


run_config = Box2DConfig()
