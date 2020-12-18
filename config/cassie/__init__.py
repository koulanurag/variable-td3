import gym
from core.config import BaseConfig
from core.env_wrapper import MultiStepWrapper


class CassieConfig(BaseConfig):
    def __init__(self):
        super(CassieConfig, self).__init__(max_env_steps=int(1e6),
                                           start_step=int(25e3),
                                           lr=3e-4,
                                           replay_memory_capacity=int(1e6),
                                           fixed_action_repeat=1,
                                           test_interval_steps=5000)

    def new_game(self, seed=None, save_video=False, video_dir_path=None, uid=None):
        import gym_cassie
        env = gym.make(self.env_name)
        env.reward_range = None
        env.close = lambda: None
        env.action_space.low = -1 * np.ones(env.action_space.low.shape)
        env.action_space.high = np.ones(env.action_space.high.shape)
        return MultiStepWrapper(env)


run_config = CassieConfig()
