import mix.utils_mix.distributed_info as dist_info
import torch
import os
import random
import numpy as np
from datetime import datetime
from mix.utils.file_io import PathManager
from mix.utils.logger import setup_logger
#from mix.utils.collect_env import collect_env_info
from mix.utils_mix.collect_running_env import collect_env_info
import argparse
import sys


def default_argument_parser(
        epilog=None):  # TODO(jinliang): rename: parse_argument()
    """

    :return:
    """
    if epilog is None:  # TODO(jinliang):copy
        epilog = """
        MIX framework running example:
        single machine:
        ${sys.argv[0]} --num-gpus 8 --config-file cfg.py MODEL.WEIGHTS /path/weight.pth

        multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """
    parser = argparse.ArgumentParser(
        epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--config-file', metavar='FILE', help='train config file path')
    parser.add_argument(
        '--resume-from', default='', help='resume from the checkpoint file')
    parser.add_argument(
        '--eval-only', action='store_true', help='just run evaluation')
    parser.add_argument(
        '--build-submit',
        action='store_true',
        help='generate submission results')
    parser.add_argument(
        '--num-gpus', type=int, default=1, help='number of gpus *per_machine')
    parser.add_argument(
        '--num-machines',
        type=int,
        default=1,
        help='total number of machines used')
    parser.add_argument(
        '--machine-rank',
        type=int,
        default=0,
        help='the rank of current machine(unique per machine)')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--options',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config file options through the command line')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    port = 2**14 + hash(os.getuid() % 2**14)  # TODO(jinliang):copy -> modify
    parser.add_argument(
        '--dist-url',
        default='tcp://127.0.0.1:{}'.format(port),
        help='initialization URL for pytorch distributed backend. See '
        'https://pytorch.org/docs/stable/distributed.html for details.',
    )

    return parser


def default_setup(args, cfg):  # DODO(jinliang):modify
    output_dir = cfg.work_dir
    if output_dir and dist_info.is_main_process():
        PathManager.mkdirs(output_dir)

    rank = dist_info.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank, name='MIX')

    logger.info('Current environment information : \n{}'.format(
        collect_env_info()))
    logger.info('Command line args: \n{}'.format(args))
    if hasattr(args, 'config_file') and args.config_file != '':
        logger.info('{} file content:\n{}'.format(
            args.config_file,
            PathManager.open(args.config_file, 'r').read()))

    logger.info('full config file content:\n{}'.format(cfg))

    if dist_info.is_main_process() and output_dir:
        cfg_path = os.path.join(output_dir, 'config.json')
        logger.info('full config file saved to {}'.format(cfg_path))

    if cfg.seed < 0:
        seed = None
    else:
        seed = cfg.seed + rank
    seed_all_rng(seed)

    torch.backends.cudnn.benchmark = False


def seed_all_rng(seed=None):  # TODO(jinliang): rename set_all_rng_seed
    if seed is None:
        dt = datetime.now()
        seed = int(dt.strftime('%S%f'))

    np.random.seed(seed)
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
