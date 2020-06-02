import gym
from gym.spaces import Box
import numpy as np
from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper


class CassieWrapper(MultiStepWrapper):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self, env):
        super(CassieWrapper, self).__init__(env)

        self._step_count = 0
        self._max_steps = 15000

    def step(self, action, action_repeat_n=1):
        assert action_repeat_n >= 1, 'action repeat should be atleast 1'
        self._step_count += action_repeat_n
        obs, reward, done, info = self.env.step(action, repeat=action_repeat_n)

        done = done or (self._step_count >= self._max_steps)
        return obs, reward, done, info

    def reset(self):
        self._step_count = 0
        return super(CassieWrapper, self).reset()

    def close(self):
        return None


class CassieConfig(BaseConfig):
    def __init__(self):
        super(CassieConfig, self).__init__(max_env_steps=int(1e6),
                                           start_step=int(25e3),
                                           lr=3e-4,
                                           replay_memory_capacity=int(1e6),
                                           fixed_action_repeat=10,
                                           test_interval_steps=5000,
                                           action_repeat_set=[10, 20, 50, 100, 200])

    def new_game(self, seed=None, save_video=False, video_dir_path=None, uid=None):
        env = self.env_factory(self.env_name)()
        env.action_space = Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        env.observation_space = Box(low=-1.0, high=1.0, shape=(42,), dtype=np.float32)
        env.reward_range = None

        env.close = lambda: None
        return CassieWrapper(env)

    @staticmethod
    def env_factory(path, state_est=False, clock_based=True, **kwargs):
        from functools import partial

        """
        Returns an *uninstantiated* environment constructor.
        Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
        this allows us to pass their constructors to Ray remote functions instead 
        (since the gym registry isn't shared across ray subprocesses we can't simply 
        pass gym.make() either)
        Note: env.unwrapped.spec is never set, if that matters for some reason.
        """
        if path in ['Cassie-v0']:
            from cassie import CassieEnv

            if path == 'Cassie-v0':
                env_fn = partial(CassieEnv, simrate=1, clock_based=clock_based, state_est=state_est)
            else:
                raise Exception("Cassie Env Unrecognized!")
            return env_fn
        else:
            raise NotImplementedError('{} Env is not implemented'.format(path))


run_config = CassieConfig()
