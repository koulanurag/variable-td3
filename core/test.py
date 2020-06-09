import pickle
import os
import numpy as np
import torch
from .model import TD3Network
from .env_wrapper import MultiStepWrapper
from typing import NamedTuple
from .utils import write_gif


class TestOutput(NamedTuple):
    score: float
    avg_repeat: float


def _test(id: int, env: MultiStepWrapper, model: TD3Network, render: bool = False, recording_path=None):
    episode_rewards = []
    action_repeats = []

    state = env.reset()
    done = False
    episode_images = []

    while not done:
        # get action
        state = torch.FloatTensor(state).unsqueeze(0)
        action = model.actor(state)
        repeat_q = model.critic_1(state, action)
        repeat_idx = repeat_q.argmax(1).item()

        action = action.data.cpu().numpy()[0]
        repeat = model.action_repeats[repeat_idx]
        action_repeats.append(repeat)

        for _ in range(repeat):
            if render:
                img = env.render(mode='rgb_array')
                episode_images.append(img)

            # step
            state, reward, done, info = env.step(action)
            episode_rewards.append(reward)

            if done:
                break

    if render:
        write_gif(episode_images, action_repeats, episode_rewards,
                  os.path.join(recording_path, 'ep_{}.gif'.format(id)))

    return sum(episode_rewards), action_repeats


def test(env: MultiStepWrapper, model: TD3Network, episodes: int, device='cpu',
         render: bool = False, save_test_data: bool = False, save_path=None, recording_path=None):
    model.to(device)
    model.eval()

    test_data = []
    for ep_i in range(episodes):
        test_data.append(_test(ep_i, env, model, render, recording_path))

    test_score, repeat_counts = zip(*test_data)

    if save_test_data:
        pickle.dump((test_score, repeat_counts), open(save_path, 'wb'))
    return TestOutput(np.array(test_score).mean(), np.mean([np.mean(x) for x in repeat_counts]))
