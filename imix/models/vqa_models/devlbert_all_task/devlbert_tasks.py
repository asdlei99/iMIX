# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import json
from io import open
from .utils import MultiTaskStopOnPlateau
import yaml
from easydict import EasyDict as edict
from ..base_model import BaseModel
from .task_utils import compute_score_with_logits
import torch
import torch.nn as nn
from imix.models.builder import VQA_MODELS
from torchvision.models import resnet
import torch.utils.model_zoo as model_zoo
from mmcv.ops import RoIAlign


@VQA_MODELS.register_module()
class DEVLBERT(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()

        self.config = config = kwargs['params']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(config['yml_path'], 'r') as f:
            task_cfg = edict(yaml.safe_load(f))
        self.task_cfg = task_cfg

        torch.backends.cudnn.deterministic = True

        if config['baseline']:
            from pytorch_transformers.modeling_bert import BertConfig
            from .basebert import BaseBertForVLTasks
        else:
            from ..vilbert.vilbert import BertConfig
            from ..vilbert.vilbert import VILBertForVLTasks

        task_names = []
        task_lr = []
        for i, task_id in enumerate(config['tasks'].split('-')):
            task = 'TASK' + task_id
            name = task_cfg[task]['name']
            task_names.append(name)
            task_lr.append(task_cfg[task]['lr'])

        base_lr = min(task_lr)
        loss_scale = {}
        task_ids = []
        for i, task_id in enumerate(config['tasks'].split('-')):
            task = 'TASK' + task_id
            loss_scale[task] = task_lr[i] / base_lr
            task_ids.append(task)

        bert_weight_name = json.load(open(config['bert_name_json']))

        bertconfig = BertConfig.from_json_file(config['config_file'])

        default_gpu = True

        if config['visual_target'] == 0:
            bertconfig.v_target_size = 1601
            bertconfig.visual_target = config['visual_target']
        else:
            bertconfig.v_target_size = 2048
            bertconfig.visual_target = config['visual_target']

        if config['task_specific_tokens']:
            bertconfig.task_specific_tokens = True

        # task_ave_iter = {}
        self.task_stop_controller = {}
        # for task_id, num_iter in task_ids:  # task_num_iters.items():
        for task_id in task_ids:
            '''
            task_ave_iter[task_id] = int(task_cfg[task]['num_epoch'] * num_iter *
                                        config['train_iter_multiplier'] /
                                        task_cfg[task]['num_epoch'])  # config['total_epochs'])
            '''
            self.task_stop_controller[task_id] = MultiTaskStopOnPlateau(
                mode='max',
                patience=1,
                continue_threshold=0.005,
                cooldown=1,
                threshold=0.001,
            )
        '''
        task_ave_iter_list = sorted(task_ave_iter.values())
        median_num_iter = task_ave_iter_list[-1]
        num_train_optimization_steps = (
            median_num_iter * \
                config['total_epochs'] // config['gradient_accumulation_steps']
        )
        '''
        num_labels = config['num_labels']
        # num_labels = max(
        #    [dataset.num_labels for dataset in task_datasets_train.values()])

        if config['dynamic_attention']:
            bertconfig.dynamic_attention = True
        if 'roberta' in config['bert_model']:
            bertconfig.model = 'roberta'

        if config['baseline']:
            self.model = BaseBertForVLTasks.from_pretrained(
                config['from_pretrained'],
                config=bertconfig,
                num_labels=num_labels,
                default_gpu=default_gpu,
            )
        else:
            self.model = VILBertForVLTasks.from_pretrained(
                config['from_pretrained'],
                config=bertconfig,
                num_labels=num_labels,
                default_gpu=default_gpu,
            )

        if config['freeze'] != -1:
            bert_weight_name_filtered = []
            for name in bert_weight_name:
                if 'embeddings' in name:
                    bert_weight_name_filtered.append(name)
                elif 'encoder' in name:
                    layer_num = name.split('.')[2]
                    if int(layer_num) <= config['freeze']:
                        bert_weight_name_filtered.append(name)

            for key, value in dict(self.model.named_parameters()).items():
                if key[12:] in bert_weight_name_filtered:
                    value.requires_grad = False

            if default_gpu:
                print('filtered weight')
                print(bert_weight_name_filtered)

        # warmpu_steps = config['warmup'] * num_train_optimization_steps

        self.lr_reduce_list = [5, 7]
        self.global_step = 0
        # start_epoch = 0
        '''
        if config['resume_file'] != "" and os.path.exists(config['resume_file']):
            checkpoint = torch.load(config['resume_file'], map_location="cpu")
            new_dict = {}
            for attr in checkpoint["model_state_dict"]:
                if attr.startswith("module."):
                    new_dict[attr.replace("module.", "", 1)] = checkpoint[
                        "model_state_dict"
                    ][attr]
                else:
                    new_dict[attr] = checkpoint["model_state_dict"][attr]
            model.load_state_dict(new_dict)
            warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            global_step = checkpoint["global_step"]
            start_epoch = int(checkpoint["epoch_id"]) + 1
            self.task_stop_controller = checkpoint["task_stop_controller"]
            tbLogger = checkpoint["tb_logger"]
            del checkpoint
        '''
        '''
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        '''
        self.task_iter_train = {name: None for name in task_ids}
        self.task_count = {name: 0 for name in task_ids}
        self.task_ids = task_ids
        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=True, final_dim=2048)

    def run_one_time(self, task_id, data):
        params = self.get_image_and_text_features(task_id, data)

        (vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction,
         vision_logit, linguisic_prediction, linguisic_logit,
         _) = self.model(params['question'], params['features'], params['spatials'], params['segment_ids'],
                         params['input_mask'], params['image_mask'], params['co_attention_mask'], params['task_tokens'])

        target = params['target']
        batch_size = params['batch_size']
        multiple_choice_ids = params['multiple_choice_ids']
        num_options = params['num_options']

        if self.task_cfg[task_id]['type'] == 'VL-classifier':
            batch_score = compute_score_with_logits(vil_prediction, target).sum()
            pred = vil_prediction

        elif self.task_cfg[task_id]['type'] == 'VL-classifier-GQA':
            batch_score = compute_score_with_logits(vil_prediction_gqa, target).sum()
            pred = vil_prediction_gqa

        elif self.task_cfg[task_id]['type'] == 'VL-logit':
            vil_logit = vil_logit.view(batch_size, num_options)
            _, preds = torch.max(vil_logit, 1)
            batch_score = float((preds == target).sum())
            pred = vil_logit

        elif self.task_cfg[task_id]['type'] == 'V-logit':
            _, select_idx = torch.max(vision_logit, dim=1)
            select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
            batch_score = float(torch.sum(select_target > 0.5))
            pred = vision_logit

        elif self.task_cfg[task_id]['type'] == 'V-logit-mc':
            vision_logit = vision_logit[:, 101:]
            vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
            vision_logit = vision_logit.unsqueeze(2)
            _, preds = torch.max(vision_logit, dim=1)
            _, target = torch.max(target, dim=1)
            batch_score = float((preds == target).sum())
            pred = vision_logit

        elif self.task_cfg[task_id]['type'] == 'VL-binary-classifier':
            batch_score = compute_score_with_logits(vil_binary_prediction, target).sum()
            pred = vil_binary_prediction

        elif self.task_cfg[task_id]['type'] == 'VL-tri-classifier':
            batch_score = compute_score_with_logits(vil_tri_prediction, target).sum()
            pred = vil_tri_prediction

        output_dict = {
            'scores': pred,
            'target': target,
            'batch_score': batch_score,
            'batch_size': batch_size,
        }

        return output_dict

    def forward_train(self, data, **kwargs):
        iterId = kwargs['cur_iter']
        epochId = kwargs['cur_epoch']
        step = kwargs['inner_iter']

        torch.autograd.set_detect_anomaly(True)
        first_task = True
        model_output = {}
        for task_id in self.task_ids:
            is_forward = False
            if (not self.task_stop_controller[task_id].in_stop) or (iterId % self.config['train_iter_gap'] == 0):
                is_forward = True

            if is_forward:
                output_dict = self.run_one_time(task_id, data)
                output_dict['batch_score'] /= output_dict['batch_size']

                model_output[task_id] = {
                    'scores': output_dict['scores'],
                    'target': output_dict['target'],
                    'batch_score': output_dict['batch_score'],
                }

                if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                    """if config['fp16']:

                    lr_this_step = config[learning_rate] * warmup_linear(
                        global_step / num_train_optimization_steps,
                        config[warmup_proportio]n,
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_this_step
                    """
                    '''
                    if first_task and (
                        global_step < warmpu_steps
                        or config['lr_scheduler'] == "warmup_linear"
                    ):
                        warmup_scheduler.step()
                    '''
                    if first_task:
                        self.global_step += 1
                        first_task = False

        # if "cosine" in config['lr_scheduler'] and global_step > warmpu_steps:
        #    lr_scheduler.step()
        '''
        if config['lr_scheduler'] == "automatic":
            lr_scheduler.step(sum(val_scores.values()))
            logger.info("best average score is %3f" % lr_scheduler.best)
        elif config['lr_scheduler'] == "mannul":
            lr_scheduler.step()
        '''
        if epochId in self.lr_reduce_list:
            for task_id in self.task_ids:
                # reset the task_stop_controller once the lr drop
                self.task_stop_controller[task_id]._reset()

        # now only one task
        return model_output[task_id]

    def forward_test(self, data, **kwargs):
        # test now does not support **kwargs
        if isinstance(self.task_ids, list):
            task_id = self.task_ids[0]
        else:
            task_id = self.task_ids
        torch.autograd.set_detect_anomaly(True)
        model_output = {}

        output_dict = self.run_one_time(task_id, data)

        model_output[task_id] = {
            'batch_score': output_dict['batch_score'],
            'batch_size': output_dict['batch_size'],
        }
        # print('batch_score', output_dict['batch_score'])
        '''
        # update the multi-task scheduler.
        self.task_stop_controller[task_id].step(
            tbLogger.getValScore(task_id))
        score = tbLogger.showLossVal(task_id, task_stop_controller)
        '''

        # now only one task
        return model_output[task_id]

    def get_image_and_text_features(self, task_id, data):
        device = self.device
        '''
        if self.task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        self.task_iter_train[task_id] = iter(task_dataloader_train[task_id])

        self.task_count[task_id] += 1
        # get the batch
        batch = self.task_iter_train[task_id].next()
        '''
        # batch = tuple(t.cuda(device=device, non_blocking=True) for k, t in data.items())
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in data)

        # data['questions_embeddings'].shape  4,4,20,768
        # data['questions_masks'].shape 4,4,20
        # data['questions_obj_tags'].shape 4,4,20
        # data['answers_embeddings'].shape 4,4,20,768
        # data['answers_masks'].shape 4,4,20
        # data['answers_obj_tags'].shape 4,4,20
        # data['label'].shape 4
        # data['segms'].shape 4,20, 14, 14
        # data['objects'].shape 4,20
        # data['boxes'].shape 4,20,4
        # data['image'].shape 4,3,384,768
        # data['max_num'].shape 4
        # data['bbox_num'].shape 4

        # origin list shape
        # 0:6,4,100,2048   features
        # 1:6,4,100,5   spatials
        # 2:6,4,100  image_mask
        # 3:6,4,30   question
        # 4:6  target
        # 5:6,4,30  input_mask
        # 6:6,4,30  segment_ids
        # 7:6,4,100,30  co_attention_mask
        # 8:6   question_id
        # if task_id == 'TASK4' or task_id == 'TASK17':
        #     (features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids,
        #      co_attention_mask, question_id) = (batch)
        # elif task_id == 'TASK5' or task_id == 'TASK6':
        #     (question, input_mask, questions_obj_tags, answers_embeddings, answers_masks,
        #      answers_obj_tags, target, segment_ids, objects, spatials, features, max_num, bbox_num) = (batch)
        # else:
        #     (features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask,
        #      question_id) = (batch)
        if task_id == 'TASK4' or task_id == 'TASK17':
            (features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids,
             co_attention_mask, question_id) = (
                 batch)
        else:
            (features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask,
             question_id) = (
                 batch)

        num_options = None
        batch_size = features.size(0)
        if self.task_cfg[task_id]['process'] in ['dialog']:
            max_num_bbox = features.size(1)
            nround = question.size(1)
            num_options = question.size(2)
            rbatch_size = batch_size * nround
            question = question.view(rbatch_size, question.size(2), question.size(3))
            target = target.view(-1)
            input_mask = input_mask.view(rbatch_size, input_mask.size(2), input_mask.size(3))
            segment_ids = segment_ids.view(rbatch_size, segment_ids.size(2), segment_ids.size(3))
            co_attention_mask = co_attention_mask.view(
                rbatch_size,
                co_attention_mask.size(2),
                co_attention_mask.size(3),
                co_attention_mask.size(4),
            )

            features = (
                features.unsqueeze(1).unsqueeze(1).expand(batch_size, nround, num_options, max_num_bbox,
                                                          2048).contiguous().view(-1, max_num_bbox, 2048))
            spatials = (
                spatials.unsqueeze(1).unsqueeze(1).expand(batch_size, nround, num_options, max_num_bbox,
                                                          5).contiguous().view(-1, max_num_bbox, 5))
            image_mask = (
                image_mask.unsqueeze(1).expand(batch_size, nround, num_options,
                                               max_num_bbox).contiguous().view(-1, max_num_bbox))

            question = question.view(-1, question.size(2))
            input_mask = input_mask.view(-1, input_mask.size(2))
            segment_ids = segment_ids.view(-1, segment_ids.size(2))
            co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))
            batch_size = rbatch_size

        elif self.task_cfg[task_id]['process'] in ['expand']:
            max_num_bbox = features.size(1)
            num_options = question.size(1)
            features = (
                features.unsqueeze(1).expand(batch_size, num_options, max_num_bbox,
                                             2048).contiguous().view(-1, max_num_bbox, 2048))
            spatials = (
                spatials.unsqueeze(1).expand(batch_size, num_options, max_num_bbox,
                                             5).contiguous().view(-1, max_num_bbox, 5))
            image_mask = (
                image_mask.unsqueeze(1).expand(batch_size, num_options,
                                               max_num_bbox).contiguous().view(-1, max_num_bbox))
            question = question.view(-1, question.size(2))
            input_mask = input_mask.view(-1, input_mask.size(2))
            segment_ids = segment_ids.view(-1, segment_ids.size(2))
            co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))

        elif self.task_cfg[task_id]['process'] in ['retrieval']:
            max_num_bbox = features.size(1)
            num_options = question.size(1)
            features = features.view(-1, features.size(2), features.size(3))
            spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
            image_mask = image_mask.view(-1, image_mask.size(2))
            question = question.view(-1, question.size(2))
            input_mask = input_mask.view(-1, input_mask.size(2))
            segment_ids = segment_ids.view(-1, segment_ids.size(2))
            co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))

        elif self.task_cfg[task_id]['process'] in ['nlvr']:
            batch_size = features.size(0)
            max_num_bbox = features.size(1)
            num_options = question.size(1)
            features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
            spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
            image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
            question = question.repeat(1, 2)
            question = question.view(batch_size * 2, int(question.size(1) / 2))
            input_mask = input_mask.repeat(1, 2)
            input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
            segment_ids = segment_ids.repeat(1, 2)
            segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
            co_attention_mask = co_attention_mask.view(
                batch_size * 2,
                int(co_attention_mask.size(1) / 2),
                co_attention_mask.size(2),
            )

        task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
        multiple_choice_ids = None
        return {
            'question': question,
            'features': features,
            'spatials': spatials,
            'segment_ids': segment_ids,
            'input_mask': input_mask,
            'image_mask': image_mask,
            'co_attention_mask': co_attention_mask,
            'task_tokens': task_tokens,
            'target': target,
            'batch_size': batch_size,
            'multiple_choice_ids': multiple_choice_ids if task_id == 'TASK4' or task_id == 'TASK17' else None,
            'num_options': num_options,
        }


class _Context_voted_module(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_Context_voted_module, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class CVM(_Context_voted_module):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(CVM, self).__init__(
            in_channels, inter_channels=inter_channels, dimension=2, sub_sample=sub_sample, bn_layer=bn_layer)


class RegionCVM(nn.Module):

    def __init__(self, in_channels, grid=[6, 6]):
        super(RegionCVM, self).__init__()
        self.CVM = CVM(in_channels, sub_sample=True, bn_layer=False)
        self.grid = grid

    def forward(self, x):
        batch_size, _, height, width = x.size()

        input_row_list = x.chunk(self.grid[0], dim=2)

        output_row_list = []
        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = row.chunk(self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                grid = self.CVM(grid)
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)
        return output


def _load_resnet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=False)
    if pretrained:
        backbone.load_state_dict(
            model_zoo.load_url('https://s3.us-west-2.amazonaws.com/ai2-rowanz/resnet50-e13db6895d81.th'))
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    return backbone


def _load_resnet_imagenet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=pretrained)
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    backbone.layer4[0].conv2.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)

    # # Make batchnorm more sensible
    # for submodule in backbone.modules():
    #     if isinstance(submodule, torch.nn.BatchNorm2d):
    #         submodule.momentum = 0.01

    return backbone


def pad_sequence(sequence, lengths):
    """
      :param sequence: ['\'sum b, .....] sequence
      :param lengths: [b1, b2, b3...] that sum to '\'sum b
      :return: [len(lengths), maxlen(b), .....] tensor
      """
    output = sequence.new_zeros(len(lengths), max(lengths), *sequence.shape[1:])
    start = 0
    for i, diff in enumerate(lengths):
        if diff > 0:
            output[i, :diff] = sequence[start:(start + diff)]
        start += diff
    return output


class Flattener(torch.nn.Module):

    def __init__(self):
        """Flattens last 3 dimensions to make it only batch size, -1."""
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class SimpleDetector(nn.Module):

    def __init__(self, pretrained=True, average_pool=True, semantic=True, final_dim=1024, layer_fix=True):
        """
        :param average_pool: whether or not to average pool the representations
        :param pretrained: Whether we need to load from scratch
        :param semantic: Whether or not we want to introduce the mask and the class label early on (default Yes)
        """
        super(SimpleDetector, self).__init__()
        USE_IMAGENET_PRETRAINED = True
        # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
        backbone = _load_resnet_imagenet(pretrained=pretrained) if USE_IMAGENET_PRETRAINED else _load_resnet(
            pretrained=pretrained)
        self.pre_backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2
        self.cvm_2 = RegionCVM(in_channels=128 * 4, grid=[6, 6])
        self.layer3 = backbone.layer3
        self.cvm_3 = RegionCVM(in_channels=256 * 4, grid=[4, 4])
        self.roi_align = RoIAlign(
            (7, 7) if USE_IMAGENET_PRETRAINED else (14, 14), spatial_scale=1 / 16, sampling_ratio=0)
        if semantic:
            self.mask_dims = 32
            self.object_embed = torch.nn.Embedding(num_embeddings=81, embedding_dim=128)
            self.mask_upsample = torch.nn.Conv2d(
                1, self.mask_dims, kernel_size=3, stride=2 if USE_IMAGENET_PRETRAINED else 1, padding=1, bias=True)
        else:
            self.object_embed = None
            self.mask_upsample = None

        self.layer4 = backbone.layer4
        self.cvm_4 = RegionCVM(in_channels=512 * 4, grid=[1, 1])
        after_roi_align = []

        self.final_dim = final_dim
        if average_pool:
            after_roi_align += [nn.AvgPool2d(7, stride=1), Flattener()]

        self.after_roi_align = torch.nn.Sequential(*after_roi_align)

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048 + (128 if semantic else 0), final_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.regularizing_predictor = torch.nn.Linear(2048, 81)

        for m in self.pre_backbone.modules():
            for p in m.parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.layer2.apply(set_bn_fix)
        self.layer3.apply(set_bn_fix)
        self.layer4.apply(set_bn_fix)
        if layer_fix:
            for m in self.layer2.modules():
                for p in m.parameters():
                    p.requires_grad = False
            for m in self.layer3.modules():
                for p in m.parameters():
                    p.requires_grad = False
            for m in self.layer4.modules():
                for p in m.parameters():
                    p.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        box_mask: torch.LongTensor,
        classes: torch.Tensor = None,
        segms: torch.Tensor = None,
    ):
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :return: object reps [batch_size, max_num_objects, dim]
        """

        images = self.pre_backbone(images)
        images = self.layer2(images)
        images = self.cvm_2(images)
        images = self.layer3(images)
        images = self.cvm_3(images)
        images = self.layer4(images)
        img_feats = self.cvm_4(images)
        box_inds = box_mask.nonzero()
        assert box_inds.shape[0] > 0
        rois = torch.cat((
            box_inds[:, 0, None].type(boxes.dtype),
            boxes[box_inds[:, 0], box_inds[:, 1]],
        ), 1)

        # Object class and segmentation representations
        roi_align_res = self.roi_align(img_feats.float(), rois.float())
        if self.mask_upsample is not None:
            assert segms is not None
            segms_indexed = segms[box_inds[:, 0], None, box_inds[:, 1]] - 0.5
            roi_align_res[:, :self.mask_dims] += self.mask_upsample(segms_indexed)

        post_roialign = self.after_roi_align(roi_align_res)

        # Add some regularization, encouraging the model to keep giving decent enough predictions
        obj_logits = self.regularizing_predictor(post_roialign)
        obj_labels = classes[box_inds[:, 0], box_inds[:, 1]]
        cnn_regularization = F.cross_entropy(obj_logits, obj_labels, reduction='mean')[None]

        feats_to_downsample = post_roialign if self.object_embed is None else torch.cat(
            (post_roialign, self.object_embed(obj_labels)), -1)
        roi_aligned_feats = self.obj_downsample(feats_to_downsample)

        # Reshape into a padded sequence - this is expensive and annoying but easier to implement and debug...
        obj_reps = pad_sequence(roi_aligned_feats, box_mask.sum(1).tolist())
        return {
            'obj_reps_raw': post_roialign,
            'obj_reps': obj_reps,
            'obj_logits': obj_logits,
            'obj_labels': obj_labels,
            'cnn_regularization_loss': cnn_regularization
        }
