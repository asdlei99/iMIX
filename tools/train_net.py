# import sys
# import os
#
# sys.path.append(os.path.abspath('.'))

from imix.engine.imix_engine import imixEngine
from imix.engine.organizer import Organizer
from imix.utils_imix.imix_checkpoint import imixCheckpointer
from imix.utils_imix.default_argument import default_argument_parser, default_setup
from imix.utils_imix.launch import launch as ddp_launch

from imix.utils_imix.config import Config as iMIX_cfg


def del_some_args(args):
    del_args = ['seed', 'work_dir', 'load_from', 'resume_from']
    for name in del_args:
        if not getattr(args, name, None):
            delattr(args, name)


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
    cfg = iMIX_cfg.fromfile(args.config_file)
    del_some_args(args)
    merge_args_to_cfg(args, cfg)
    default_setup(args, cfg)

    return cfg


def test(cfg):
    assert cfg.get('load_from', None), '--load-from is empty '

    model = Organizer.build_model(cfg)
    check_pointer = imixCheckpointer(model, save_dir=cfg.work_dir)
    check_pointer.resume_or_load(cfg.load_from, resume=False)

    result = Organizer.test(cfg, model)

    # if 'test' in cfg.test_datasets:
    #     Organizer.build_test_result(cfg, model)
    # else:
    #     result = Organizer.test(cfg, model)
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
    ddp_launch(
        run_fn=main,
        gpus=args.gpus,
        machines=args.machines,
        master_addr=args.master_addr,
        master_port=args.master_port,
        run_fn_args=(args, ))
