from torch import nn
import torch
# from imix.utils.registry import Registry, build_from_cfg
from imix.utils_imix.registry import Registry, build_from_cfg
from imix.data.parallel.distributed import MMDistributedDataParallel
from imix.data.parallel.data_parallel import MMDataParallel

EMBEDDING = Registry('embedding')
ENCODER = Registry('encoder')
BACKBONES = Registry('backbone')
COMBINE_LAYERS = Registry('combine_layers')
HEADS = Registry('head')
LOSSES = Registry('loss')
VQA_MODELS = Registry('vqa_models')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_embedding(cfg):
    """Build neck."""
    return build(cfg, EMBEDDING)


def build_encoder(cfg):
    """Build roi extractor."""
    return build(cfg, ENCODER)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_combine_layer(cfg):
    """Build shared head."""
    return build(cfg, COMBINE_LAYERS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    from imix.evaluation.evaluator_mix1 import build as list_build
    return list_build(cfg, LOSSES)

    # return build(cfg, LOSSES)


def build_vqa_models(cfg):
    """Build vqa_models."""
    return build(cfg.model, VQA_MODELS)


def build_model(cfg):
    """Build models based on different input type."""
    model = build_vqa_models(cfg)  # TODO(jinliang)

    # # put model on gpus
    # if cfg.distributed:
    #     find_unused_parameters = cfg.get('find_unused_parameters', False)
    #     # Sets the `find_unused_parameters` parameter in
    #     # torch.nn.parallel.DistributedDataParallel
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False,
    #         find_unused_parameters=find_unused_parameters)
    # else:
    #     # model = MMDataParallel(
    #     #     model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    #     model = model.cuda(cfg.gpu_ids[0])

    model.to(torch.device(cfg.model_device))

    return model
