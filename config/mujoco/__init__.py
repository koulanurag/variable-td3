import gym

from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper


class MujocoConfig(BaseConfig):
    def __init__(self):
        super(MujocoConfig, self).__init__(max_env_steps=int(1e6),
                                           start_step=int(25e3),
                                           replay_memory_capacity=100000,
                                           fixed_action_repeat=1)

    def new_game(self, seed=None, save_video=False, video_dir_path=None, uid=None):
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=video_dir_path, force=True, video_callable=lambda episode_id: True, uid=uid)
        return MultiStepWrapper(env)


run_config = MujocoConfig()
