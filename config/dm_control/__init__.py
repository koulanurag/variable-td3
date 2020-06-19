import gym

from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper


class DMControlWrapper(gym.Wrapper):
    def __init__(self, env):
        super(DMControlWrapper, self).__init__(env)
        self.observation_space = env.observation_space['observations']

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs['observations'], reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs['observations']

    def render(self, mode='human', *args, **kwargs):
        return self.env.render(mode, *args, use_opencv_renderer=True, **kwargs)


class DmControlConfig(BaseConfig):
    def __init__(self):
        super(DmControlConfig, self).__init__(max_env_steps=int(1e6),
                                              start_step=int(25e3),
                                              lr=3e-4,
                                              replay_memory_capacity=int(1e6),
                                              fixed_action_repeat=1,
                                              test_interval_steps=5000)

    def new_game(self, seed=None):
        env = gym.make('dm2gym:' + self.env_name, environment_kwargs={'flat_observation': True})
        env = DMControlWrapper(env)

        if seed is not None:
            env.seed(seed)

        return MultiStepWrapper(env)


run_config = DmControlConfig()
