# TODO(jinliang):jinliang_imitate

# import sys
#
# sys.path.append('/home/jinliang/code/Mix/mix')

import os
from mix.utils.config import Config
from mix.engine.mix_engine import MixEngine
from mix.engine.organizer import Organizer
from mix.utils.mix_checkpoint import MixCheckpointer
# from mix.utils.default_argument import default_argument_parser, default_setup
from mix.utils_mix.default_argument import default_argument_parser, default_setup
# from mix.utils.launch import launch
from mix.utils_mix.launch import launch as ddp_launch

from mix.utils_mix.config import Config as mix_config

# import argparse
# import torch
# from mix.utils.file_io import PathManager
# from mix.utils import comm
# from mix.utils.logger import setup_logger
# from mix.utils.env import seed_all_rng
# from mix.utils.collect_env import collect_env_info
# from mix.evaluation import verify_results
# import warnings

# def merge_args_to_cfg(cfg, args):  # TODO(jinliang):jinliang_copy
#     for k, v in vars(args).items():
#         if k == 'work_dir' and v is None:
#             if cfg.get(k, None) is None:
#                 cfg.work_dir = os.path.join(
#                     os.path.splitext(os.path.basename(args.config))[0],
#                     './work_dir')
#         else:
#             cfg[k] = v


def del_some_args(args):
  if args.seed is None:
    del args.seed
  if args.work_dir is None:
    del args.work_dir


def merge_args_to_cfg(args, cfg):
  for k, v in vars(args).items():
    cfg[k] = v


def init_set(args):
  """
    This function initalizes related parmeters
    1. command line parameters
    2. read args.config file
    3. Itegration of command line parameters and arg.config file parameters
    4. setting logging.
    """

  # cfg = Config.fromfile(args.config_file)
  # cfg_mix = mix_config.fromfile(args.config_file)
  # del cfg
  # cfg = cfg_mix

  cfg = mix_config.fromfile(args.config_file)
  del_some_args(args)
  merge_args_to_cfg(args, cfg)
  default_setup(args, cfg)

  return cfg


def test(cfg):
  model = Organizer.build_model(cfg)
  mix_ck = MixCheckpointer(model, save_dir=cfg.work_dir)
  mix_ck.resume_or_load(cfg.load_from, resume=False)

  result = []
  # Organizer.build_test_result(cfg, model)
  if 'test' in cfg.test_datasets:
    Organizer.build_test_result(cfg, model)
  else:
    result = Organizer.test(cfg, model)
  # if comm.is_main_process():
  #     verify_results(cfg, result)
  return result


def train(cfg):
  mix_trainer = MixEngine(cfg)
  return mix_trainer.train()


def main(args):
  cfg = init_set(args)
  if cfg.eval_only:
    return test(cfg)
  else:
    return train(cfg)


if __name__ == '__main__':
  args = default_argument_parser().parse_args()
  print('Command line Args:', args)
  # launch(
  #     main,
  #     args.num_gpus,
  #     num_machines=args.num_machines,
  #     machine_rank=args.machine_rank,
  #     dist_url=args.dist_url,
  #     args=(args,),
  # )

  ddp_launch(
      run_fn=main,
      nproc_per_node=args.nproc_per_node,
      nnodes=args.nnodes,
      master_addr=args.master_addr,
      master_port=args.master_port,
      run_fn_args=(args,))
