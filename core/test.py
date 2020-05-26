import pickle

import numpy as np
import torch
from .model import TD3Network
from .env_wrapper import MultiStepWrapper
from typing import NamedTuple


class TestOutput(NamedTuple):
    score: float
    avg_repeat: float


def _test(env: MultiStepWrapper, model: TD3Network, render: bool = False):
    episode_reward = 0
    action_repeats = []

    state = env.reset()
    done = False

    while not done:
        if render:
            env.render()

        # get action
        state = torch.FloatTensor(state).unsqueeze(0)
        action = model.actor(state)
        repeat_q = model.critic_1(state, action)
        repeat_idx = repeat_q.argmax(1).item()

        action = action.data.cpu().numpy()[0]
        repeat = model.action_repeats[repeat_idx]

        # step
        state, reward, done, info = env.step(action, repeat)  # Step
        episode_reward += reward
        action_repeats.append(repeat)

    return episode_reward, action_repeats


def test(env: MultiStepWrapper, model: TD3Network, episodes: int, device='cpu',
         render: bool = False, save_test_data: bool = False, save_path=None):
    model.to(device)
    model.eval()

    test_data = []
    for ep_i in range(episodes):
        test_data.append(_test(env, model, render))

    test_score, repeat_counts = zip(*test_data)

    if save_test_data:
        pickle.dump((test_score, repeat_counts), open(save_path, 'wb'))
    return TestOutput(np.array(test_score).mean(), np.mean([np.mean(x) for x in repeat_counts]))
