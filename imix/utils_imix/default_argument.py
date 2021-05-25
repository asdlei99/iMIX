import imix.utils_imix.distributed_info as dist_info
import torch
import os
import random
from imix.utils_imix.file_io import PathManager

from imix.utils_imix.logger import setup_logger
from imix.utils_imix.collect_running_env import collect_env_info
import argparse
import json
from imix.utils_imix.config import set_imix_work_dir, seed_all_rng
import pprint


def default_argument_parser(epilog=None):
    if epilog is None:
        epilog = """
        imix framework running example:
        single machine:
        ${sys.argv[0]} --gpus 8 --config-file cfg.py MODEL.WEIGHTS /path/weight.pth

        multiple machines:
        (machine0)$ {sys.argv[0]} --gpus 8 --node-rank 0 --machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --gpus 8 --node-rank 1 --machines 2 --dist-url <URL> [--other-flags]
        """
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--config-file', metavar='FILE', help='train config file path')
    parser.add_argument('--resume-from', default=None, help='resume from the checkpoint file')
    parser.add_argument('--load-from', default=None, help='load from the checkpoint file')
    parser.add_argument('--eval-only', action='store_true', help='just run evaluation')
    parser.add_argument('--build-submit', action='store_true', help='generate submission results')
    parser.add_argument('--gpus', type=int, default=1, help='the number of gpus on each machine ')
    parser.add_argument('--machines', type=int, default=1, help='the total number of machine to use')
    parser.add_argument('--node-rank', type=int, default=0, help='the rank of current node(unique per machine)')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--master-port',
        default=2**14 + hash(random.randint(0, 2**14)),
        type=int,
        help='it is the free port of mast node(rank 0) and is used for communication in distributed training')
    parser.add_argument(
        '--master-addr', default='tcp://127.0.0.1', type=str, help='the IP address of mast node(rank 0)')

    return parser


def default_setup(args, cfg):
    output_dir = cfg.work_dir
    set_imix_work_dir(output_dir)
    if output_dir and dist_info.is_main_process():
        PathManager.mkdirs(output_dir)

    rank = dist_info.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank, name='imix')
    logger.info('Current environment information : \n{}'.format(collect_env_info()))
    logger.info('Command line args: \n {}'.format(args))
    if hasattr(args, 'config_file') and args.config_file != '':
        logger.info('{} file content:\n{}'.format(args.config_file, PathManager.open(args.config_file, 'r').read()))

    logger.info('full config file content: ')
    pprint.pprint({k: v for k, v in cfg.items()})

    if dist_info.is_main_process() and output_dir:
        cfg_path = os.path.join(output_dir, 'config.json')
        with open(cfg_path, 'w') as f:
            f.write(json.dumps({k: v for k, v in cfg.items()}, indent=4, separators=(',', ':')))
            logger.info('full config file saved to {}'.format(cfg_path))

    seed = getattr(cfg, 'seed', None)
    seed_all_rng(seed=None if seed is None else seed + rank)

    if not (hasattr(cfg, 'eval_only') and getattr(cfg, 'eval_only', False)):
        torch.backends.cudnn.benchmark = getattr(cfg, 'CUDNN_BENCHMARK', False)
