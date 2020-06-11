import argparse
import logging.config
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from core.test import test
from core.train import train
from core.utils import init_logger, make_results_dir

if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='TD3 with variable action repeat')
    parser.add_argument('--env', required=True,
                        help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--case', required=True, choices=['dm_control', 'mujoco', 'box2d', 'classic_control', 'cassie'],
                        help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--opr', required=True, choices=['train', 'test'])
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='no cuda usage (default: %(default)s)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Renders the environment (default: %(default)s)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overrides past results (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: %(default)s)')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Use Weight and bias visualization lib (default: %(default)s)')
    parser.add_argument('--action_repeat_mode', choices=['fixed', 'variable'], default='variable',
                        help='Mode of action repeat (default: %(default)s)')
    parser.add_argument('--fixed_action_repeat', type=int, default=None,
                        help='Action Repeat (default: %(default)s)')
    parser.add_argument('--test_episodes', type=int, default=1,
                        help='Evaluation episode count (default: %(default)s)')

    # Process arguments
    args = parser.parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'

    # seeding random iterators
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # import corresponding configuration , neural networks and envs
    if args.case == 'classic_control':
        from config.classic_control import run_config
    elif args.case == 'box2d':
        from config.box2d import run_config
    elif args.case == 'mujoco':
        from config.mujoco import run_config
    elif args.case == 'dm_control':
        from config.dm_control import run_config
    elif args.case == 'cassie':
        from config.cassie import run_config
    else:
        raise Exception('Invalid --case option.')

    # set config as per arguments
    run_config.set_config(args)
    log_base_path = make_results_dir(run_config.exp_path, args)

    # set-up logger
    init_logger(log_base_path)
    logging.getLogger('root').info('cmd args:{}'.format(' '.join(sys.argv[1:])))  # log command line arguments.

    try:
        if args.opr == 'train':
            if args.use_wandb:
                import wandb

                wandb.init(group=args.case + ':' + args.env, project="variable-td3",
                           config=run_config.get_hparams(), sync_tensorboard=True)

            summary_writer = SummaryWriter(run_config.exp_path, flush_secs=60 * 1)  # flush every 1 minutes
            train(run_config, summary_writer)
            summary_writer.flush()
            summary_writer.close()

            if args.use_wandb:
                wandb.join()
        elif args.opr == 'test':
            model_path = run_config.model_path
            assert model_path, 'model not found: {}'.format(model_path)

            model = run_config.get_uniform_network()
            model = model.to('cpu')
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

            if args.render and args.case == 'mujoco':
                # Ref: https://github.com/openai/mujoco-py/issues/390
                from mujoco_py import GlfwContext
                GlfwContext(offscreen=True)

            env = run_config.new_game()
            test_score, test_repeat_counts = test(env, model, args.test_episodes,
                                                  device='cpu', render=args.render,
                                                  save_test_data=True, save_path=run_config.test_data_path,
                                                  recording_path=run_config.recording_path)
            env.close()

            logging.getLogger('test').info('Test Score: {}'.format(test_score))
        else:
            raise NotImplementedError('"--opr {}" is not implemented ( or not valid)'.format(args.opr))

    except Exception as e:
        logging.getLogger('root').error(e, exc_info=True)

    logging.shutdown()
