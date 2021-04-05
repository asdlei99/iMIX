# TODO(jinliang):jinliang_imitate

# import sys
#
# sys.path.append('/home/jinliang/code/imix/imix')

import os
from imix.utils.config import Config
from imix.engine.imix_engine import imixEngine
from imix.engine.organizer import Organizer
from imix.utils.imix_checkpoint import imixCheckpointer
# from imix.utils.default_argument import default_argument_parser, default_setup
from imix.utils_imix.default_argument import default_argument_parser, default_setup
# from imix.utils.launch import launch
from imix.utils_imix.launch import launch as ddp_launch

from imix.utils_imix.config import Config as imix_config

# import argparse
# import torch
# from imix.utils.file_io import PathManager
# from imix.utils import comm
# from imix.utils.logger import setup_logger
# from imix.utils.env import seed_all_rng
# from imix.utils.collect_env import collect_env_info
# from imix.evaluation import verify_results
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
    # cfg_imix = imix_config.fromfile(args.config_file)
    # del cfg
    # cfg = cfg_imix

    cfg = imix_config.fromfile(args.config_file)
    del_some_args(args)
    merge_args_to_cfg(args, cfg)
    default_setup(args, cfg)

    return cfg


def test(cfg):
    model = Organizer.build_model(cfg)
    imix_ck = imixCheckpointer(model, save_dir=cfg.work_dir)
    imix_ck.resume_or_load(cfg.load_from, resume=False)

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
    imix_trainer = imixEngine(cfg)
    return imix_trainer.train()


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
        run_fn_args=(args, ))
