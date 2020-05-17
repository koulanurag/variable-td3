import itertools
import logging
import random

import numpy as np
import torch
from torch.distributions import Normal
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from core.utils import get_epsilon
from .config import BaseConfig
from .replay_memory import ReplayMemory
from .test import test

train_logger = logging.getLogger('train')
test_logger = logging.getLogger('train_eval')


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def update_params(model, target_model, critic_optimizer, policy_optimizer, memory, updates, config, writer):
    # Sample a batch from memory
    batch = memory.sample(batch_size=config.batch_size)

    # pre-process batch
    state_batch = torch.FloatTensor(batch.state).to(config.device)
    next_state_batch = torch.FloatTensor(batch.next_state).to(config.device)
    action_batch = torch.FloatTensor(batch.action).to(config.device)
    action_repeat_batch = torch.FloatTensor(batch.action_repeat).to(config.device).long().unsqueeze(1)
    reward_batch = torch.FloatTensor(batch.reward).to(config.device).unsqueeze(1)
    mask_batch = torch.FloatTensor(batch.mask).to(config.device).unsqueeze(1)

    # Compute Targets for Q values
    with torch.no_grad():
        # Select next action according to target policy:
        next_action = target_model.actor(next_state_batch)
        noise_dist = Normal(torch.tensor([0.0]), torch.tensor([config.policy_noise]))
        noise = noise_dist.sample(next_action.shape).squeeze(-1)
        noise = noise.clamp(-config.noise_clip, config.noise_clip)
        next_action = next_action + noise
        next_action = config.clip_action(next_action)

        q1_next_target = target_model.critic_1(next_state_batch, next_action)
        q2_next_target = target_model.critic_2(next_state_batch, next_action)
        min_q_next_target = torch.min(q1_next_target, q2_next_target)
        max_q_repeat_target = min_q_next_target.max(dim=1)[0].unsqueeze(1)
        next_q_value = reward_batch + (mask_batch * config.gamma * max_q_repeat_target)

    # Compute Loss for  Q_values
    q1 = model.critic_1(state_batch, action_batch)
    q2 = model.critic_2(state_batch, action_batch)

    q1_loss = MSELoss()(q1.gather(1, action_repeat_batch), next_q_value)
    q2_loss = MSELoss()(q2.gather(1, action_repeat_batch), next_q_value)

    # Update critic network
    critic_optimizer.zero_grad()
    (q1_loss + q2_loss).backward()
    critic_optimizer.step()

    # Compute Loss for Policy
    action = model.actor(state_batch)
    q1 = model.critic_1(state_batch, action)
    policy_loss = -q1.max(1)[0].mean()

    # Update policy network
    if updates % config.policy_delay == 0:
        critic_optimizer.zero_grad()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

    # Update target network
    soft_update(target_model, model, config.tau)

    return q1_loss.item(), q2_loss.item(), policy_loss.item()


def train(config: BaseConfig, writer: SummaryWriter):
    memory = ReplayMemory(config.replay_memory_capacity)

    # create networks & optimizer
    model = config.get_uniform_network().to(config.device)
    target_model = config.get_uniform_network().to(config.device)
    target_model.load_state_dict(model.state_dict())
    test_model = config.get_uniform_network().to(config.device)

    critic_optimizer = Adam([{'params': model.critic_1.parameters()},
                             {'params': model.critic_2.parameters()}], lr=config.lr)
    policy_optimizer = Adam(model.actor.parameters(), lr=config.lr)

    total_env_steps = 0
    updates = 0
    best_test_score = float('-inf')
    env = config.new_game()
    for i_episode in itertools.count(1):

        done = False
        episode_steps, episode_reward = 0, 0
        epsilon = get_epsilon(config.max_epsilon, config.min_epsilon, total_env_steps, config.max_env_steps)
        state = env.reset()

        while not done:
            with torch.no_grad():
                # noisy action
                action = model.actor(torch.FloatTensor(state).unsqueeze(0))
                noise = Normal(torch.tensor([0.0]), torch.tensor([config.exploration_noise]))
                action = action + noise.sample(action.shape).squeeze(-1)
                action = config.clip_action(action)

                # epsilon-greedy repeat
                repeat_q = model.critic_1(torch.FloatTensor(state).unsqueeze(0), action)
                if np.random.rand() <= epsilon:
                    repeat_idx = random.randrange(len(model.action_repeats))
                else:
                    repeat_idx = repeat_q.argmax(1).item()

            # step
            action = action.data.cpu().numpy()[0]
            repeat = model.action_repeats[repeat_idx]
            next_state, reward, done, info = env.step(action, repeat)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            mask = 1 if (('TimeLimit.truncated' in info) and info['TimeLimit.truncated']) else float(not done)

            # Add to memory
            memory.push(state, action, repeat_idx, reward, next_state, mask)

            episode_steps += repeat
            total_env_steps += repeat
            episode_reward += reward
            state = next_state

            # update network
            if len(memory) > config.batch_size:
                critic_1_loss, critic_2_loss, policy_loss = 0, 0, 0
                for i in range(config.updates_per_step * repeat):
                    loss_data = update_params(model, target_model, critic_optimizer,
                                              policy_optimizer, memory, updates, config, writer)
                    critic_1_loss += loss_data[0]
                    critic_2_loss += loss_data[1]
                    policy_loss += loss_data[2]

                    updates += 1

                critic_1_loss /= (config.updates_per_step * repeat)
                critic_2_loss /= (config.updates_per_step * repeat)
                policy_loss /= (config.updates_per_step * repeat)

                # Log
                writer.add_scalar('train/critic_1_loss', critic_1_loss, total_env_steps)
                writer.add_scalar('train/critic_2_loss', critic_1_loss, total_env_steps)
                writer.add_scalar('train/policy_loss', policy_loss, total_env_steps)

        # log episode data
        writer.add_scalar('data/eps_reward', episode_reward, total_env_steps)
        writer.add_scalar('data/eps_steps', episode_steps, total_env_steps)
        writer.add_scalar('data/episodes', i_episode, total_env_steps)
        writer.add_scalar('data/epsilon', epsilon, total_env_steps)
        writer.add_scalar('data/updates', updates, total_env_steps)

        _msg = '#{} train score:{} eps steps: {} total steps: {} updates : {}'
        _msg = _msg.format(i_episode, round(episode_reward, 2), episode_steps, total_env_steps, updates)
        train_logger.info(_msg)

        # Test
        if i_episode % config.test_interval == 0:
            test_model.load_state_dict(model.state_dict())
            test_score, avg_action_repeats = test(env, test_model, config.test_episodes)
            if test_score > best_test_score:
                torch.save(test_model.state_dict(), config.best_model_path)

            # Test Log
            writer.add_scalar('test/score', test_score, total_env_steps)
            writer.add_scalar('test/avg_action_repeats', avg_action_repeats, total_env_steps)
            test_logger.info('#{} test score: {} avg_action_repeats:{}'.format(i_episode, test_score,
                                                                               avg_action_repeats))
        # save model
        if i_episode % config.save_model_freq == 0:
            torch.save(model.state_dict(), config.model_path)

        # check if max. env steps reached.
        if total_env_steps > config.max_env_steps:
            train_logger.info('max env. steps reached!!')
            break

    # save the last updated model
    torch.save(model.state_dict(), config.model_path)
