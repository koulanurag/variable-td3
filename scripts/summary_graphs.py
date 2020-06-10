"""
Usage:
Step 1: python summary_graphs.py --logdir=../results/classic_control --opr extract_summary
Step 2: python summary_graphs.py --logdir=../results/classic_control --opr plot

Why 2-step process? : Sometimes , you may want to re-do the plotting with some changes for more beautification
as the first step takes a while.
"""

import argparse
import os
import pickle
import numpy as np
from collections import defaultdict
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorboard.backend.event_processing import event_accumulator as ea

TEST_TAG = {'ref': 'test/score'}
AVG_ACTION_REPEAT_TAG = {'ref': 'test/avg_action_repeats'}

SINGLE_GRAPH_WIDTH, SINGLE_GRAPH_HEIGHT = 350, 350


def extract_summaries(logdir: str):
    """ extracts and pickles only relevant scalars

    :param logdir: Path to the directory having event logs
    """
    # Collect data : we recognize all files which have tfevents
    scalars_info = defaultdict(dict)

    for root, dirs, files in os.walk(logdir):
        game_name = root.split('-v')[0].split('/')[-1]
        event_files = [x for x in files if 'tfevents' in x]
        if len(event_files) > 0:
            assert len(event_files) == 1, 'only one tf file allowed per experiment.'

            if game_name not in scalars_info:
                scalars_info[game_name] = {'fixed': {}, 'variable': {}}

            event_path = os.path.join(root, event_files[0])
            acc = ea.EventAccumulator(event_path)
            acc.Reload()  # load data

            if 'fixed' in root:
                repeat_mode = 'fixed'
                repeat_count = root.split('action_repeat_')[1].split('/')[0]
                if repeat_count not in scalars_info[game_name]['fixed']:
                    scalars_info[game_name]['fixed'][repeat_count] = {'test': {'seed': {}},
                                                                      'avg_action_repeat': {'seed': {}}}
                dest = scalars_info[game_name]['fixed'][repeat_count]
            elif 'variable' in root:
                repeat_mode = 'variable'
                repeat_count = ''
                if 'test' not in scalars_info[game_name]['variable']:
                    scalars_info[game_name]['variable'] = {'test': {'seed': {}},
                                                           'avg_action_repeat': {'seed': {}}}
                dest = scalars_info[game_name]['variable']
            else:
                raise NotImplementedError

            seed = root.split('seed_')[1].split('/')[0]
            x = [s.step for s in acc.Scalars(TEST_TAG['ref'])]
            y = [s.value for s in acc.Scalars(TEST_TAG['ref'])]
            dest['test']['seed'][seed] = {'x': x, 'y': y}

            x = [s.step for s in acc.Scalars(AVG_ACTION_REPEAT_TAG['ref'])]
            y = [s.value for s in acc.Scalars(AVG_ACTION_REPEAT_TAG['ref'])]
            dest['avg_action_repeat']['seed'][seed] = {'x': x, 'y': y}

            print('Processed {}, seed:{} , mode:{} , repeat: {}'.format(game_name, seed, repeat_mode, repeat_count))

    return scalars_info


def _plot(scalars_info, save_dir, column_size=3):
    """ Plots scalars using plotly. """

    # Note: Refer here for color names : https://www.w3schools.com/cssref/css_colors.asp
    candidate_colors = ['coral', 'orchid', 'palegreen', 'yellow', 'thistle',
                        'turquoise', 'chartreuse', 'darkcyan', 'darkmagenta']

    # make titles
    titles = sorted(list(scalars_info.keys()))

    # overall plot attributes
    column_size = min(column_size, len(titles))
    rows = math.ceil(len(titles) / column_size)
    cols = column_size
    test_fig = make_subplots(rows, cols, subplot_titles=titles, horizontal_spacing=0.05,
                             vertical_spacing=0.05, print_grid=True)
    action_repeat_fig = make_subplots(rows, cols, subplot_titles=titles, horizontal_spacing=0.05,
                                      vertical_spacing=0.05, print_grid=True)

    mode_colors = {}

    for game_i, game_name in enumerate(titles):
        modes = [('variable', '')]
        modes += [('fixed', repeat) for repeat in scalars_info[game_name]['fixed'].keys()]

        row, col = game_i // column_size + 1, (game_i % column_size) + 1

        for (mode, repeat) in modes:

            if mode == 'fixed':
                src = scalars_info[game_name][mode][repeat]
                trace_name = mode + '(' + repeat + ')'
            else:
                src = scalars_info[game_name][mode]
                trace_name = mode

            if (mode, repeat) in mode_colors:
                mode_color = mode_colors[(mode, repeat)]
            else:
                mode_color = candidate_colors.pop(0)
                mode_colors[(mode, repeat)] = mode_color

            for (category, fig) in [('test', test_fig), ('avg_action_repeat', action_repeat_fig)]:
                data_y = [src[category]['seed'][seed]['y'] for seed in src[category]['seed']]
                data_x = src[category]['seed'][list(src[category]['seed'].keys())[0]]['x']

                print(game_name, category, trace_name)

                mean = np.array(data_y).mean(axis=0)
                std = np.std(data_y, axis=0)

                fig.add_trace(go.Scatter(x=data_x,
                                         y=mean,
                                         name=trace_name,
                                         showlegend=game_i == len(titles) - 1,
                                         line=dict(color=mode_color)),
                              row=row, col=col)

                fig.add_trace(go.Scatter(x=data_x,
                                         y=mean - std,
                                         name=trace_name,
                                         fill=None,
                                         showlegend=False,
                                         line=dict(color=mode_color, width=0.1)),
                              row=row, col=col)

                fig.add_trace(go.Scatter(x=data_x,
                                         y=mean + std,
                                         name=trace_name,
                                         fill='tonexty',
                                         showlegend=False,
                                         line=dict(color=mode_color, width=0.1)),
                              row=row, col=col)

    # update font size of title
    for i in test_fig['layout']['annotations']:
        i['font']['size'] = 12
    for i in action_repeat_fig['layout']['annotations']:
        i['font']['size'] = 12
    test_fig.update_layout(showlegend=True, template='seaborn', legend_orientation='h',
                           legend=dict(x=0.5, y=-0.25, xanchor='center', yanchor='middle'),
                           margin=dict(l=20, r=20, b=0, t=40, pad=0), )
    action_repeat_fig.update_layout(showlegend=True, legend_orientation='h', template='seaborn',
                                    legend=dict(x=0.5, y=-0.25, xanchor='center', yanchor='middle'),
                                    margin=dict(l=20, r=20, b=0, t=40, pad=0), )

    # save plot as image
    test_img_path = os.path.join(save_dir, 'test_summary.png')
    action_repeat_img_path = os.path.join(save_dir, 'action_repeat_summary.png')
    test_fig.write_image(test_img_path,
                         width=cols * SINGLE_GRAPH_WIDTH, height=rows * SINGLE_GRAPH_HEIGHT)
    action_repeat_fig.write_image(action_repeat_img_path,
                                  width=cols * SINGLE_GRAPH_WIDTH, height=rows * SINGLE_GRAPH_HEIGHT)

    print('Summary Figure saved @ :', test_img_path)
    print('Summary Figure saved @ :', action_repeat_img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Custom script for extracting data from tf summaries and '
                                     'plotting only specific scalars  in plotly')
    parser.add_argument('--logdir', type=str, help='Path to event files', required=True)
    parser.add_argument('--opr', required=True, choices=['extract_summary', 'plot'])

    args = parser.parse_args()
    summary_dir = os.path.join(args.logdir, 'summary')
    scalar_info_path = os.path.join(summary_dir, 'scalars_info.p')
    os.makedirs(summary_dir, exist_ok=True)

    if args.opr == 'extract_summary':
        print('Please , note that it may take a while ...')
        scalars_info = extract_summaries(logdir=args.logdir)
        pickle.dump(scalars_info, open(scalar_info_path, 'wb'))
    elif args.opr == 'plot':
        scalars_info = pickle.load(open(scalar_info_path, 'rb'))
        _plot(scalars_info, summary_dir)
    else:
        raise NotImplementedError('"--opr {}" is not implemented ( or not valid)'.format(args.opr))
