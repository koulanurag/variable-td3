import itertools
import logging

import numpy as np
import torch
from torch.distributions import Normal
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from core.utils import get_epsilon
from .config import BaseConfig
from .replay_memory import ReplayMemory, BatchOutput
from .test import test

train_logger = logging.getLogger('train')
test_logger = logging.getLogger('train_eval')


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def update_params(model, target_model, critic_optimizer, policy_optimizer, memory, updates, config, writer):
    # Sample a batch from memory
    batch: BatchOutput = memory.sample(batch_size=config.batch_size)

    # pre-process batch
    state_batch = torch.FloatTensor(batch.state).to(config.device)
    next_state_batch = torch.FloatTensor(batch.next_state).to(config.device)
    next_state_mask_batch = torch.FloatTensor(batch.next_state_mask).to(config.device).bool()
    action_batch = torch.FloatTensor(batch.action).to(config.device)
    reward_batch = torch.FloatTensor(batch.reward).to(config.device)
    terminal_batch = torch.FloatTensor(batch.terminal).to(config.device)

    # Compute Loss for  Q_values
    q1 = model.critic_1(state_batch, action_batch)
    q2 = model.critic_2(state_batch, action_batch)
    q1_loss, q2_loss = torch.zeros(reward_batch.shape), torch.zeros(reward_batch.shape)

    # Compute Targets for Q values
    for repeat_i in range(len(model.action_repeats)):
        valid_next_state_batch = next_state_batch[:, repeat_i][next_state_mask_batch[:, repeat_i]]
        valid_reward_batch = reward_batch[:, repeat_i][next_state_mask_batch[:, repeat_i]]
        valid_terminal_batch = terminal_batch[:, repeat_i][next_state_mask_batch[:, repeat_i]]
        repeat_n = model.action_repeats[repeat_i]

        if len(valid_next_state_batch) > 0:
            with torch.no_grad():
                next_action = target_model.actor(valid_next_state_batch)
                noise_dist = Normal(torch.tensor([0.0]), torch.tensor([config.policy_noise]))
                noise = noise_dist.sample(next_action.shape).squeeze(-1).to(config.device)
                noise = noise.clamp(-config.noise_clip, config.noise_clip)
                next_action = next_action + noise
                next_action = config.clip_action(next_action)

                q1_next_target = target_model.critic_1(valid_next_state_batch, next_action)
                q2_next_target = target_model.critic_2(valid_next_state_batch, next_action)
                min_q_next_target = torch.min(q1_next_target, q2_next_target)
                max_q_repeat_target = min_q_next_target.max(dim=1)[0]
                next_q_value = valid_reward_batch + ((1 - valid_terminal_batch) *
                                                     (config.gamma ** repeat_n) * max_q_repeat_target)

            mse = MSELoss(reduction='none')
            q1_src = q1[:, repeat_i][next_state_mask_batch[:, repeat_i]]
            q1_loss[:, repeat_i][next_state_mask_batch[:, repeat_i]] = mse(q1_src, next_q_value)

            q2_src = q2[:, repeat_i][next_state_mask_batch[:, repeat_i]]
            q2_loss[:, repeat_i][next_state_mask_batch[:, repeat_i]] = mse(q2_src, next_q_value)

            if repeat_i == 0:
                writer.add_histogram('next_q_value', next_q_value, updates)
                writer.add_histogram('q1', q1_src, updates)

    # Update critic network
    critic_optimizer.zero_grad()
    q1_loss = (q1_loss.sum(dim=1) / next_state_mask_batch.sum(dim=1)).mean()
    q2_loss = (q2_loss.sum(dim=1) / next_state_mask_batch.sum(dim=1)).mean()
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
    env = config.new_game(seed=config.seed)

    for i_episode in itertools.count(1):

        done = False
        episode_steps, episode_reward, epsiode_repeats = 0, 0, []

        state = env.reset()
        while not done:
            epsilon = get_epsilon(config.max_epsilon, config.min_epsilon, total_env_steps, config.max_env_steps)

            if total_env_steps < config.start_step:
                action = env.action_space.sample()
                repeat_n = np.random.choice(model.action_repeats)
            else:
                with torch.no_grad():
                    # noisy action
                    state = torch.FloatTensor(state).unsqueeze(0).to(config.device)
                    action = model.actor(state)
                    noise = Normal(torch.tensor([0.0]), torch.tensor([config.exploration_noise]))
                    action = action + noise.sample(action.shape).squeeze(-1).to(config.device)
                    action = config.clip_action(action)

                    # epsilon-greedy repeat
                    repeat_q = model.critic_1(state, action)
                    if np.random.rand() <= epsilon:
                        repeat_idx = np.random.randint(0, len(model.action_repeats) - 1)
                    else:
                        repeat_idx = repeat_q.argmax(1).item()

                state = state.data.cpu().numpy()[0]
                action = action.data.cpu().numpy()[0]
                repeat_n = model.action_repeats[repeat_idx]

            # step
            epsiode_repeats.append(repeat_n)
            discounted_reward_sum = 0
            next_states, rewards, terminals = [], [], []

            step = 0
            for repeat_i in range(1, repeat_n + 1):
                next_state, reward, done, info = env.step(action)
                discounted_reward_sum += (config.gamma ** repeat_i) * reward
                episode_steps += 1
                total_env_steps += 1
                episode_reward += reward
                step += 1

                if (repeat_i in model.action_repeats) or done:
                    next_states.append(next_state)
                    rewards.append(discounted_reward_sum)

                    # Ignore the "done" signal if it comes from hitting the time horizon.
                    terminal = 0 if (('TimeLimit.truncated' in info) and info['TimeLimit.truncated']) else float(done)
                    terminals.append(terminal)

                if done:
                    break

            next_state_mask = [1 for _ in range(len(next_states) - 1)] + [1]
            if len(next_states) < len(model.action_repeats):
                next_state_mask += [0 for _ in range(len(model.action_repeats) - len(next_state_mask))]

                # Note: these values will be ignored during update
                terminals += [float('-inf') for _ in range(len(model.action_repeats) - len(terminals))]
                next_states += [np.ones(next_states[-1].shape)
                                for _ in range(len(model.action_repeats) - len(next_states))]
                rewards += [float('-inf') for _ in range(len(model.action_repeats) - len(rewards))]

            # Add to memory
            memory.push(state, action, rewards, next_states, next_state_mask, terminals)

            state = next_states[-1]

            # update network
            if len(memory) > config.batch_size:
                critic_1_loss, critic_2_loss, policy_loss = 0, 0, 0
                update_count = config.updates_per_step * step
                for i in range(update_count):
                    loss = update_params(model, target_model, critic_optimizer,
                                         policy_optimizer, memory, updates, config, writer)
                    critic_1_loss += loss[0]
                    critic_2_loss += loss[1]
                    policy_loss += loss[2]

                    updates += 1

                critic_1_loss /= update_count
                critic_2_loss /= update_count
                policy_loss /= update_count

                # Log
                writer.add_scalar('train/critic_1_loss', critic_1_loss, total_env_steps)
                writer.add_scalar('train/critic_2_loss', critic_2_loss, total_env_steps)
                writer.add_scalar('train/policy_loss', policy_loss, total_env_steps)

        # log episode data
        writer.add_scalar('data/eps_reward', episode_reward, total_env_steps)
        writer.add_scalar('data/eps_steps', episode_steps, total_env_steps)
        writer.add_scalar('data/eps_repeats', np.array(epsiode_repeats).mean(), total_env_steps)
        writer.add_scalar('data/episodes', i_episode, total_env_steps)
        writer.add_scalar('data/epsilon', epsilon, total_env_steps)
        writer.add_scalar('train/updates', updates, total_env_steps)
        for name, W in model.named_parameters():
            writer.add_histogram('network_weights' + '/' + name, W.data.cpu().numpy(), total_env_steps)

        _msg = '#{} train score:{} eps steps: {} total steps: {} updates : {}'
        _msg = _msg.format(i_episode, round(episode_reward, 2), episode_steps, total_env_steps, updates)
        train_logger.info(_msg)

        # Test
        if i_episode % config.test_interval == 0:
            test_model.load_state_dict(model.state_dict())
            test_output = test(env, test_model, config.test_episodes)
            if test_output.score > best_test_score:
                torch.save(test_model.state_dict(), config.best_model_path)

            # Test Log
            writer.add_scalar('test/score', test_output.score, total_env_steps)
            writer.add_scalar('test/avg_action_repeats', test_output.avg_repeat, total_env_steps)
            test_logger.info('#{} test score: {} avg_action_repeats:{}'.format(i_episode,
                                                                               test_output.score,
                                                                               test_output.avg_repeat))
        # save model
        if i_episode % config.save_model_freq == 0:
            torch.save(model.state_dict(), config.model_path)

        # check if max. env steps reached.
        if total_env_steps > config.max_env_steps:
            train_logger.info('max env. steps reached!!')
            break

    # save the last updated model
    torch.save(model.state_dict(), config.model_path)
