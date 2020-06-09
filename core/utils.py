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


def write_gif(episode_images, action_repeats, episode_rewards, gif_path):
    assert len(episode_images) == len(episode_rewards)

    import plotly.graph_objects as go
    from io import BytesIO
    from PIL import Image

    rep_fig = go.Figure(data=go.Scatter(x=[], y=[]))
    rep_fig.update_layout(
        title="Action Repeat Count",
        xaxis_title="Time Step",
        yaxis_title="Action Repeat",
    )

    score_fig = go.Figure(data=go.Scatter(x=[], y=[]))
    score_fig.update_layout(
        title="Episode Score",
        xaxis_title="Time Step",
        yaxis_title="Score",
    )

    episode_stats = []
    total_reward = 0
    _obs = Image.fromarray(episode_images[0])
    width, height = _obs.width, _obs.height
    step_i = 0

    for repeat in action_repeats:
        # update repeat count figure
        rep_fig['data'][0]['x'] += tuple([step_i])
        rep_fig['data'][0]['y'] += tuple([repeat])
        repeat_img = Image.open(BytesIO(rep_fig.to_image(format="png",
                                                         width=width, height=height)))

        pop_i = 0
        while pop_i <= repeat and step_i < len(episode_images):

            # obs
            obs = Image.fromarray(episode_images[step_i])

            # update score figure.
            total_reward += episode_rewards[step_i]
            score_fig['data'][0]['x'] += tuple([step_i])
            score_fig['data'][0]['y'] += tuple([total_reward])
            score_img = Image.open(BytesIO(score_fig.to_image(format="png",
                                                              width=width, height=height)))

            # combine repeat image + actual obs + score image
            overall_img = Image.new('RGB', (repeat_img.width + obs.width + score_img.width, repeat_img.height))
            overall_img.paste(obs, (0, 0))
            overall_img.paste(repeat_img, (obs.width, 0))
            overall_img.paste(score_img, (obs.width + repeat_img.width, 0))
            episode_stats.append(overall_img)

            # incr counters
            step_i += 1
            pop_i += 1

    assert total_reward == sum(episode_rewards)
    assert step_i == len(episode_images)

    # save as gif
    episode_stats[0].save(gif_path, save_all=True, append_images=episode_stats[1:], optimize=False, loop=1)
