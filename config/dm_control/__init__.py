import gym

from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper


class DMControlWrapper(gym.Wrapper):
    def __init__(self, env):
        super(DMControlWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs['observations'], reward, done, info

    def reset(self):
        obs = super().reset()
        return obs['observations']

    def render(self, mode='human', *args, **kwargs):
        super().render(use_opencv_renderer=True)


class DmControlConfig(BaseConfig):
    def __init__(self):
        super(DmControlConfig, self).__init__()

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, eval=False):
        env = gym.make('dm2gym:' + self.env_name, environment_kwargs={'flat_observation': True})
        env = DMControlWrapper(env)

        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return MultiStepWrapper(env)


run_config = DmControlConfig()
