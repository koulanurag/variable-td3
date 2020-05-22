import gym

from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper


class CassieConfig(BaseConfig):
    def __init__(self):
        super(CassieConfig, self).__init__()

    def new_game(self, seed=None, save_video=False, video_dir_path=None, uid=None):
        env = self.env_factory(self.env_name)

        return MultiStepWrapper(env)

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
                env_fn = partial(CassieEnv, clock_based=clock_based, state_est=state_est)
            else:
                raise Exception("Cassie Env Unrecognized!")
            return env_fn

        spec = gym.envs.registry.spec(path)
        _kwargs = spec._kwargs.copy()
        _kwargs.update(kwargs)

        try:
            if callable(spec._entry_point):
                cls = spec._entry_point(**_kwargs)
            else:
                cls = gym.envs.registration.load(spec._entry_point)
        except AttributeError:
            if callable(spec.entry_point):
                cls = spec.entry_point(**_kwargs)
            else:
                cls = gym.envs.registration.load(spec.entry_point)

        return partial(cls, **_kwargs)


run_config = CassieConfig()
