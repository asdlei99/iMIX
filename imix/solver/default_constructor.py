import warnings

import torch
import json
from torch.nn import GroupNorm, LayerNorm
from .builder import OPTIMIZER_BUILDERS, OPTIMIZERS
# from ..utils.registry import build_from_cfg, Registry
from ..utils_imix.registry import build_from_cfg
# from ..utils.parrots_wrapper import _BatchNorm, _InstanceNorm
from ..utils.misc import is_list_of
# import imix.utils.comm as comm
# import imix.utils_imix.distributed_info as comm
from torch.optim.optimizer import required
from typing import Callable, Iterable, Tuple
from torch.optim import Optimizer
import math
import logging


@OPTIMIZER_BUILDERS.register_module()
class DefaultOptimizerConstructor:
    """Default constructor for optimizers.

    By default each parameter share the same optimizer settings, and we
    provide an argument ``paramwise_cfg`` to specify parameter-wise settings.
    It is a dict and may contain the following fields:

    - ``custom_keys`` (dict): Specified parameters-wise settings by keys. If
      one of the keys in ``custom_keys`` is a substring of the name of one
      parameter, then the setting of the parameter will be specified by
      ``custom_keys[key]`` and other setting like ``bias_lr_mult`` etc. will
      be ignored. It should be noted that the aforementioned ``key`` is the
      longest key that is a substring of the name of the parameter. If there
      are multiple matched keys with the same length, then the key with lower
      alphabet order will be chosen.
      ``custom_keys[key]`` should be a dict and may contain fields ``lr_mult``
      and ``decay_mult``. See Example 2 below.
    - ``bias_lr_mult`` (float): It will be multiplied to the learning
      rate for all bias parameters (except for those in normalization
      layers).
    - ``bias_decay_mult`` (float): It will be multiplied to the weight
      decay for all bias parameters (except for those in
      normalization layers and depthwise conv layers).
    - ``norm_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of normalization
      layers.
    - ``dwconv_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of depthwise conv
      layers.
    - ``bypass_duplicate`` (bool): If true, the duplicate parameters
      would not be added into optimizer. Default: False.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are

                - `type`: class name of the optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.
        paramwise_cfg (dict, optional): Parameter-wise options.

    Example 1:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> paramwise_cfg = dict(norm_decay_mult=0.)
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)

    Example 2:
        >>> # assume model have attribute model.backbone and model.cls_head
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, weight_decay=0.95)
        >>> paramwise_cfg = dict(custom_keys={
                '.backbone': dict(lr_mult=0.1, decay_mult=0.9)})
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)
        >>> # Then the `lr` and `weight_decay` for model.backbone is
        >>> # (0.01 * 0.1, 0.95 * 0.9). `lr` and `weight_decay` for
        >>> # model.cls_head is (0.01, 0.95).
    """

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        if not isinstance(optimizer_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict', f'but got {type(optimizer_cfg)}')
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        self.base_lr = optimizer_cfg.get('lr', None)
        self.base_wd = optimizer_cfg.get('weight_decay', None)
        self._validate_cfg()

    def _validate_cfg(self):
        if not isinstance(self.paramwise_cfg, dict):
            raise TypeError('paramwise_cfg should be None or a dict, ' f'but got {type(self.paramwise_cfg)}')

        if 'custom_keys' in self.paramwise_cfg:
            if not isinstance(self.paramwise_cfg['custom_keys'], dict):
                raise TypeError('If specified, custom_keys must be a dict, '
                                f'but got {type(self.paramwise_cfg["custom_keys"])}')
            if self.base_wd is None:
                for key in self.paramwise_cfg['custom_keys']:
                    if 'decay_mult' in self.paramwise_cfg['custom_keys'][key]:
                        raise ValueError('base_wd should not be None')

        # get base lr and weight decay
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in self.paramwise_cfg or 'norm_decay_mult' in self.paramwise_cfg
                or 'dwconv_decay_mult' in self.paramwise_cfg):
            if self.base_wd is None:
                raise ValueError('base_wd should not be None')

    def _is_in(self, param_group, param_group_list):
        assert is_list_of(param_group_list, dict)
        param = set(param_group['params'])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group['params']))

        return not param.isdisjoint(param_set)

    def add_params(self, params, module, prefix=''):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)

        # special rules for norm layers and depth-wise conv layers
        # is_norm = isinstance(module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_norm = isinstance(module, (GroupNorm, LayerNorm))
        is_dwconv = (isinstance(module, torch.nn.Conv2d) and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                warnings.warn(f'{prefix} is duplicate. It is skipped since ' f'bypass_duplicate={bypass_duplicate}')
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    break
            if not is_custom:
                # bias_lr_mult affects all bias parameters except for norm.bias
                if name == 'bias' and not is_norm:
                    param_group['lr'] = self.base_lr * bias_lr_mult
                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group['weight_decay'] = self.base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group['weight_decay'] = self.base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == 'bias':
                        param_group['weight_decay'] = self.base_wd * bias_decay_mult
            params.append(param_group)

        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(params, child_mod, prefix=child_prefix)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            # if hasattr(model, "get_optimizer_parameters"):
            #     optimizer_cfg['params'] = model.get_optimizer_parameters(optimizer_cfg['lr'],
            #                                                            optimizer_cfg['training_encoder_lr_multiply'])
            #     # if comm.get_world_size() > 1:
            #     #     optimizer_cfg['params'] = model.module.get_optimizer_parameters(optimizer_cfg['lr'],
            #     #                                                                     optimizer_cfg[
            #     #                                                                     'training_encoder_lr_multiply'])
            #     # else:
            #     #     optimizer_cfg['params'] = model.get_optimizer_parameters(optimizer_cfg['lr'],
            #     #                                                              optimizer_cfg[
            #     #                                                                  'training_encoder_lr_multiply'])
            #
            #     optimizer_cfg.pop('training_encoder_lr_multiply')
            # else:
            #     optimizer_cfg['params'] = model.parameters()
            optimizer_cfg['params'] = model.parameters()
            optimizer_cfg.pop('training_encoder_lr_multiply')
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params = []
        self.add_params(params, model)
        optimizer_cfg['params'] = params

        # return build_from_cfg(optimizer_cfg, OPTIMIZERS)
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@OPTIMIZER_BUILDERS.register_module()
class BertOptimizerConstructor(DefaultOptimizerConstructor):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        self.language_weights_file = paramwise_cfg.pop('language_weights_file', None)
        super().__init__(optimizer_cfg=optimizer_cfg, paramwise_cfg=paramwise_cfg)
        self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    def modify_params(self, params, optimizer_cfg, module, prefix=''):
        langauge_weights = self.load_language_weight(file=self.language_weights_file)

        for key, value in dict(module.named_parameters()).items():
            if value.requires_grad:
                if key in langauge_weights:
                    lr = optimizer_cfg['lr']
                else:
                    lr = optimizer_cfg['image_lr']

                if any(nd in key for nd in self.no_decay):
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0}]

                if not any(nd in key for nd in self.no_decay):
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.01}]

    @staticmethod
    def load_language_weight(file):
        with open(file) as f:
            return json.load(f)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        params = []
        self.modify_params(params, self.optimizer_cfg, model)
        optimizer_cfg['params'] = params

        optimizer_cfg.pop('image_lr')
        optimizer_cfg.pop('training_encoder_lr_multiply')

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@OPTIMIZERS.register_module()
class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.

    Params:
        betas: Adams b1. Default: 0.9
               Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self,
                 params,
                 lr=required,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0.01,
                 max_grad_norm=1.0):
        b1 = betas[0]
        b2 = betas[1]
        if lr is not required and lr < 0.0:
            raise ValueError('Invalid learning rate: {} - should be >= 0.0'.format(lr))
        if not 0.0 <= b1 < 1.0:
            raise ValueError('Invalid b1 parameter: {} - should be in [0.0, 1.0['.format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError('Invalid b2 parameter: {} - should be in [0.0, 1.0['.format(b2))
        if not eps >= 0.0:
            raise ValueError('Invalid epsilon value: {} - should be >= 0.0'.format(eps))
        defaults = dict(lr=lr, b1=b1, b2=b2, e=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # LXRT: grad is clipped outside.
                # Add grad clipping
                # if group['max_grad_norm'] > 0:
                #     clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                update_with_lr = group['lr'] * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


@OPTIMIZERS.register_module()
class TansformerAdamW(Optimizer):
    """Implements Adam algorithm with weight decay fix as introduced in
    `Decoupled Weight Decay Regularization.

    <https://arxiv.org/abs/1711.05101>`__.

    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError('Invalid learning rate: {} - should be >= 0.0'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter: {} - should be in [0.0, 1.0['.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter: {} - should be in [0.0, 1.0['.format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {} - should be >= 0.0'.format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1**state['step']
                    bias_correction2 = 1.0 - beta2**state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    """Linearly increases learning rate over `warmup`*`t_total` (as provided to
    BertAdam) training steps.

    Learning rate is 1. afterwards.
    """
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    """Specifies a triangular learning rate schedule where peak is reached at
    `warmup`*`t_total`-th (as provided to BertAdam) training step.

    After `t_total`-th training step, learning rate is zero.
    """
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


@OPTIMIZERS.register_module()
class LXMERT_BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.

    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self,
                 params,
                 lr=required,
                 warmup=-1,
                 t_total=-1,
                 schedule='warmup_linear',
                 b1=0.9,
                 b2=0.999,
                 e=1e-6,
                 weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError('Invalid learning rate: {} - should be >= 0.0'.format(lr))
        if schedule not in SCHEDULES:
            raise ValueError('Invalid schedule parameter: {}'.format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError('Invalid warmup: {} - should be in [0.0, 1.0[ or -1'.format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError('Invalid b1 parameter: {} - should be in [0.0, 1.0['.format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError('Invalid b2 parameter: {} - should be in [0.0, 1.0['.format(b2))
        if not e >= 0.0:
            raise ValueError('Invalid epsilon value: {} - should be >= 0.0'.format(e))
        defaults = dict(
            lr=lr,
            schedule=schedule,
            warmup=warmup,
            t_total=t_total,
            b1=b1,
            b2=b2,
            e=e,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm)
        super(LXMERT_BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if comm.is_main_process():
            logger = logging.getLogger(__name__)

        if closure is not None:
            loss = closure()

        warned_for_t_total = False

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # LXRT: grad is clipped outside.
                # Add grad clipping
                # if group['max_grad_norm'] > 0:
                #     clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    progress = state['step'] / group['t_total']
                    lr_scheduled = group['lr'] * schedule_fct(progress, group['warmup'])
                    # warning for exceeding t_total (only active with warmup_linear
                    if group['schedule'] == 'warmup_linear' and progress > 1. and not warned_for_t_total:
                        if comm.is_main_process():
                            logger.warning("Training beyond specified 't_total' steps with schedule '{}'. "
                                           "Learning rate set to {}. Please set 't_total' of {} correctly.".format(
                                               group['schedule'], lr_scheduled, self.__class__.__name__))
                        warned_for_t_total = True
                    # end warning
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss
