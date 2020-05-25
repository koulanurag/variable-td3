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
import random
from collections import defaultdict
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from tensorboard.backend.event_processing import event_accumulator as ea

# Note: Refer here for color names : https://www.w3schools.com/cssref/css_colors.asp
TEST_TAG = {'ref': 'test/score', 'color': 'coral'}

SINGLE_GRAPH_WIDTH, SINGLE_GRAPH_HEIGHT = 350, 350


def extract_summaries(logdir: str):
    """ re-draw the tf summary events plots  using plotly

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
                scalars_info[game_name] = {'fixed': {}, 'variable': {'seed': {}}}

            event_path = os.path.join(root, event_files[0])
            acc = ea.EventAccumulator(event_path)
            acc.Reload()

            if 'fixed' in root:
                repeat_mode = 'fixed'
                repeat_count = root.split('action_repeat_')[1].split('/')[0]
                if repeat_count not in scalars_info[game_name]['fixed']:
                    scalars_info[game_name]['fixed'][repeat_count] = {'seed': {}}
                dest = scalars_info[game_name]['fixed'][repeat_count]
            elif 'variable' in root:
                repeat_mode = 'variable'
                dest = scalars_info[game_name]['variable']
            else:
                raise NotImplementedError

            seed = root.split('seed_')[1].split('/')[0]
            x = [s.step for s in acc.Scalars(TEST_TAG['ref'])]
            y = [s.value for s in acc.Scalars(TEST_TAG['ref'])]

            dest['seed'][seed] = {'x': x, 'y': y}
            print(game_name, seed, repeat_mode, len(x), len(y))

    return scalars_info


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
