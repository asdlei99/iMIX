# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function
import logging
import torch
from pytorch_transformers import BertConfig
from .modeling.modeling_bert import ImageBertForMultipleChoice, ImageBertForSequenceClassification
from .utils.task_utils import processors
from imix.models.builder import VQA_MODELS
from ..base_model import BaseModel
import sys

sys.path.insert(0, '.')

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )), ())

MODEL_CLASSES = {
    'bert': (BertConfig, ImageBertForSequenceClassification),
}

log_json = []


@VQA_MODELS.register_module()
class OSCAR_NLVR(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        args = kwargs['params']
        # Prepare GLUE task
        task_name = args.task_name.lower()
        if task_name not in processors:
            raise ValueError('Task not found: %s' % (task_name))

        # args.output_mode = output_modes[task_name]
        num_labels = args.num_labels
        logger.info('Task Name: {}, #Labels: {}'.format(task_name, num_labels))

        # Load pretrained model and tokenizer
        # if args.local_rank not in [-1, 0]:
        #     torch.distributed.barrier()
        #     # Make sure only the first process in distributed training will download model & vocab

        self.model_type = args.model_type.lower()
        config_class, model_class = MODEL_CLASSES[self.model_type]
        if args.use_pair:
            model_class = ImageBertForMultipleChoice

        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=task_name)

        # discrete code
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.code_voc = args.code_voc
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.use_layernorm = args.use_layernorm
        config.classifier = args.classifier
        config.cls_hidden_scale = args.cls_hidden_scale
        config.num_choice = args.num_choice

        self.model = model_class.from_pretrained(
            args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info('Model Parameters: {}'.format(total_params))

        self.img_feature_dim = args.img_feature_dim

    def forward_train(self, data, **kwargs):
        """Train the model."""
        # if self.fp16:
        #     try:
        #         from apex import amp
        #     except ImportError:
        #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        #     model, optimizer = amp.initialize(model, optimizer, opt_level=self.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        # if self.n_gpu > 1:
        #     model = torch.nn.DataParallel(model)
        # Distributed training (should be after apex fp16 initialization)
        # if self.local_rank != -1:
        #     model = torch.nn.parallel.DistributedDataParallel(
        #         model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        batch = tuple(t.to(self.device) for t in data)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2] if self.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
            'labels': batch[3],
            'img_feats': None if self.img_feature_dim == -1 else batch[4]
        }
        outputs = self.model(**inputs)

        logits = outputs[0]

        batch_score = logits.argmax(1) == batch[3].view(-1).sum()
        batch_size = batch[0].size(0)

        model_output = {
            'scores': logits,
            'target': batch[3],
            'batch_score': batch_score,
            'batch_size': batch_size,
        }

        return model_output

    def forward_test(self, data, **kwargs):
        # eval_task_names = ("mnli", "mnli-mm") if self.task_name == "mnli" else (self.task_name, )
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(self.device) for t in data)

        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2] if self.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
            'labels': batch[3],
            'img_feats': None if self.img_feature_dim == -1 else batch[4]
        }

        outputs = self.model(**inputs)
        logits = outputs[0]

        val, idx = logits.max(1)
        batch_score = torch.sum(idx == batch[3].view(-1))
        batch_size = batch[0].size(0)

        model_output = {
            'scores': logits,
            'target': batch[3],
            'batch_score': batch_score,
            'batch_size': batch_size,
        }

        return model_output
