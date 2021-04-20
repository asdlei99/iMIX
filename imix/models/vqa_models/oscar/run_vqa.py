# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import logging
import torch

from .modeling.modeling_bert import ImageBertForSequenceClassification
from pytorch_transformers import BertConfig

from .utils.task_utils import processors
from imix.models.builder import VQA_MODELS
from ..base_model import BaseModel
import sys

sys.path.insert(0, '.')
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, ImageBertForSequenceClassification),
}


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


@VQA_MODELS.register_module()
class OSCAR(BaseModel):

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
        config.classifier = args.classifier
        config.cls_hidden_scale = args.cls_hidden_scale
        # config.use_img_layernorm = args.use_img_layernorm

        # code_level = args.code_level
        # img_feature_type = args.img_feature_type
        # data_dir = args.data_dir
        # # load discrete code
        # if img_feature_type in ['dis_code', 'dis_code_t']:
        #     logger.info('Load discrete code from: {}'.format(data_dir))FF
        #     t_start = time.time()
        #     train_code = torch.load(os.path.join(data_dir, 'vqvae', 'train.pt'))
        #     t_end = time.time()
        #     logger.info('Load time: %.3f' % (t_end - t_start))

        #     if code_level == 'top':
        #         config.code_dim = train_code['embeddings_t'].shape[0]
        #         config.code_size = train_code['feats_top'][list(train_code['feats_top'].keys())[0]].shape[0]
        #     elif code_level == 'bottom':
        #         config.code_dim = train_code['embeddings_b'].shape[0]
        #         config.code_size = train_code['feats_bottom'][list(train_code['feats_bottom'].keys())[0]].shape[0]
        #     elif code_level == 'both':
        #         config.code_dim = train_code['embeddings_t'].shape[0] + train_code['embeddings_b'].shape[0]

        self.model = model_class.from_pretrained(
            args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        # if img_feature_type in ['dis_code', 'dis_code_t']:
        #     logger.info('Initializing the code embedding with {}'.format(code_level))
        #     if code_level == 'top':
        #         self.model.init_code_embedding(train_code['embeddings_t'].t())
        #     elif code_level == 'bottom':
        #         self.model.init_code_embedding(train_code['embeddings_b'].t())

        # if args.local_rank == 0:
        #     torch.distributed.barrier(
        #     )  # Make sure only the first process in distributed training will download model & vocab

        self.task_name = task_name
        # self.adjust_dp = args.adjust_dp
        # self.adjust_loss = args.adjust_loss
        # self.adjust_loss_epoch = args.adjust_loss_epoch
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

        # if self.adjust_dp and epoch >= 3:
        #     logger.info("change droput ratio {} to 0.3".format(self.drop_out))
        #     if hasattr(model, 'module'):
        #         model.module.dropout.p = 0.3
        #         model.module.bert.dropout.p = 0.3
        #         model.module.bert.embeddings.dropout.p = 0.3
        #     else:
        #         model.dropout.p = 0.3
        #         model.bert.dropout.p = 0.3
        #         model.bert.embeddings.dropout.p = 0.3

        # if self.adjust_loss and epoch >= self.adjust_loss_epoch:
        #     logger.info("\t change loss type from kl to bce")
        #     model.loss_type = 'bce'

        batch = tuple(t.to(self.device) for t in data)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2] if self.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
            'labels': batch[4],
            'img_feats': None if self.img_feature_dim == -1 else batch[5]
        }
        outputs = self.model(**inputs)

        logits = outputs[0]

        # if self.fp16:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()

        # batch_score = compute_score_with_logits(logits, batch[4]).sum()
        batch_score = torch.sum(compute_score_with_logits(logits, batch[4]), 1).sum()
        batch_size = batch[0].size(0)

        model_output = {
            'scores': logits,
            'target': batch[4],
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
            'labels': batch[4],
            'img_feats': None if self.img_feature_dim == -1 else batch[5]
        }

        outputs = self.model(**inputs)
        logits = outputs[0]

        batch_score = torch.sum(compute_score_with_logits(logits, batch[4]), 1).sum()
        batch_size = batch[0].size(0)

        model_output = {
            'scores': logits,
            'target': batch[4],
            'batch_score': batch_score,
            'batch_size': batch_size,
        }

        return model_output
