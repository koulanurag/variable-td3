import os
from typing import List

import torch

from core.model import TD3Network


class BaseConfig(object):

    def __init__(self,
                 max_env_steps: int,
                 start_step: int,
                 batch_size: int = 256,
                 updates_per_step: int = 1,
                 exploration_noise: float = 0.1,
                 lr: float = 1e-3,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 alpha: float = 0.2,
                 max_epsilon: float = 0.5,
                 min_epsilon: float = 0.1,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 grad_norm_clip: float = 40,
                 policy_delay: int = 2,
                 save_model_freq: int = 50,
                 replay_memory_capacity: int = 1e6,
                 action_repeat_set: List[int] = [2, 4, 8, 16, 32],
                 fixed_action_repeat=2,
                 test_interval_steps=5000,
                 test_episodes=5):

        # training
        self.max_env_steps = max_env_steps
        self.start_step = start_step
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.updates_per_step = updates_per_step
        self.grad_norm_clip = grad_norm_clip
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.alpha = alpha
        self.replay_memory_capacity = replay_memory_capacity
        self.device = 'cpu'
        self.action_repeat_mode = None
        self.seed = None
        self.save_model_freq = save_model_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.test_episodes = test_episodes

        # reward normalization
        self.reward_scale_factor = None

        # exploration
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon

        # test
        self.test_interval_steps = test_interval_steps

        # action info
        self.action_repeat_set = action_repeat_set
        self.fixed_action_repeat = fixed_action_repeat

        # env info
        self.env_name = None
        self.observation_space = None
        self.action_space = None

        # paths
        self.exp_path = None
        self.model_path = None
        self.best_model_path = None
        self.test_data_path = None

    def new_game(self, seed=None, save_video=False, video_dir_path=None, uid=None):
        raise NotImplementedError

    def get_uniform_network(self):
        action_repeats = [self.fixed_action_repeat] if self.action_repeat_mode == 'fixed' else self.action_repeat_set
        return TD3Network(self.observation_space.shape[0], self.action_space.shape[0], action_repeats,
                          hidden_dim=256, action_space=self.action_space)

    def clip_action(self, action):
        assert len(action.shape) == 2
        clamped_action = [torch.clamp(action[:, a_i].unsqueeze(1),
                                      self.action_space.low[a_i],
                                      self.action_space.high[a_i])
                          for a_i in range(action.shape[1])]
        action = torch.cat(clamped_action, dim=1)
        return action

    def get_hparams(self):
        hparams = {}
        for k, v in self.__dict__.items():
            if 'path' not in k and (v is not None):
                hparams[k] = v
        return hparams

    def set_config(self, args):
        # env info
        self.env_name = args.env
        env = self.new_game()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_repeat_mode = args.action_repeat_mode
        env.close()

        # training
        self.seed = args.seed
        self.device = args.device

        if args.fixed_action_repeat is not None:
            self.fixed_action_repeat = args.fixed_action_repeat

        # create experiment path
        self.exp_path = os.path.join(args.result_dir, args.case, args.env)

        # action repeat mode
        if self.action_repeat_mode == 'fixed':
            self.reward_scale_factor = self.fixed_action_repeat
            self.exp_path = os.path.join(self.exp_path, 'fixed', 'action_repeat_{}'.format(self.fixed_action_repeat))
        elif self.action_repeat_mode == 'variable':
            self.reward_scale_factor = np.mean(self.action_repeat_set)
            self.exp_path = os.path.join(self.exp_path, 'variable')
        else:
            raise AttributeError('action_repeat_mode : {} is not valid'.format(self.action_repeat_mode))

        # other parameters
        self.exp_path = os.path.join(self.exp_path, 'seed_{}'.format(self.seed))

        # other paths
        self.model_path = os.path.join(self.exp_path, 'model.p')
        self.best_model_path = os.path.join(self.exp_path, 'best_model.p')
        self.test_data_path = os.path.join(self.exp_path, 'test_data.p')
