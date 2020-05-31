import logging
import os
import shutil
import torch
from gym.spaces import Space


def get_epsilon(max_eps: float, min_eps: float, curr_steps: int, max_steps: int):
    epsilon = max(min_eps, max_eps - (max_eps - min_eps) * curr_steps / (0.5 * max_steps))
    return epsilon


def clip_action(action, action_space: Space):
    assert len(action.shape) == 2, 'Expected batch of actions'
    clamped_action = [torch.clamp(action[:, a_i].unsqueeze(1),
                                  action_space.low[a_i],
                                  action_space.high[a_i])
                      for a_i in range(action.shape[1])]
    action = torch.cat(clamped_action, dim=1)
    return action


def make_results_dir(exp_path, args):
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force:
            raise FileExistsError('{} is not empty. Please use --force to overwrite it'.format(exp_path))
        else:
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    return log_path


def init_logger(base_path):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['train', 'test', 'train_eval', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
