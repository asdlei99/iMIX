import argparse
import os
import warnings

import torch
from iimix.engine.iimix_engine import iimixEngine
from iimix.utils.config import Config

from imix.engine.organizer import Organizer
from imix.evaluation import verify_results
# from iopath.common.file_io import PathManager
from imix.utils import comm
from imix.utils.collect_env import collect_env_info
from imix.utils.env import seed_all_rng
from imix.utils.file_io import PathManager
from imix.utils.imix_checkpoint import imixCheckpointer
from imix.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='train multimodal')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus', type=int, help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids', type=int, nargs='+', help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--eval-only', action='store_true', help='only test if eval_only is true')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # if args.options and args.cfg_options:
    #     raise ValueError(
    #         '--options and --cfg-options cannot be both '
    #         'specified, --options is deprecated in favor of --cfg-options')
    # if args.options:
    #     warnings.warn('--options is deprecated in favor of --cfg-options')
    #     args.cfg_options = args.options

    return args


def default_setup(cfg, args):
    """Perform some basic common setups at the beginning of a job, including:

    1. Set up the imix logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.work_dir
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name='imix')
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info('Rank of current process: {}. World size: {}'.format(rank, comm.get_world_size()))
    logger.info('Environment info:\n' + collect_env_info())

    logger.info('Command line arguments: ' + str(args))
    if hasattr(args, 'config_file') and args.config_file != '':
        logger.info('Contents of args.config_file={}:\n{}'.format(args.config_file,
                                                                  PathManager.open(args.config_file, 'r').read()))

    logger.info('Running with full config:\n{}'.format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, 'config.yaml')
        with PathManager.open(path, 'w') as f:
            f.write(cfg.dump())
        logger.info('Full config saved to {}'.format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    # if not (hasattr(args, "eval_only") and args.eval_only):
    #     torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK  #TODO(jinliang) delete


def init_set():
    """
      This function initalizes related parmeters
      1. command line parameters
      2. read args.config file
      3. Itegration of command line parameters and arg.config file parameters
      4. setting logging.
      """
    args = parse_args()
    print('command line args:', args)

    cfg = Config.fromfile(args.config)

    cfg.eval_only = args.eval_only
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.distributed = False
    else:
        cfg.distributed = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join(os.path.splitext(os.path.basename(args.config))[0], './work_dir')

    default_setup(cfg, args)

    return cfg


def test(cfg):
    model = Organizer.build_model(cfg)
    imix_ck = imixCheckpointer(model, save_dir=cfg.work_dir)
    imix_ck.resume_or_load(cfg.load_from, resume=False)
    # TODO(jinliang): add to load the trained model
    result = Organizer.test(cfg, model)
    # if comm.is_main_process():
    #     verify_results(cfg, result)
    return result


def train(cfg):
    imix_trainer = imixEngine(cfg)
    return imix_trainer.train_iter()


def main():
    cfg = init_set()
    if cfg.eval_only:
        return test(cfg)
    else:
        return train(cfg)


if __name__ == '__main__':
    main()
