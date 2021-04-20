# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# TODO(jinliang):jinliang_imitate
import copy
import itertools
import json
import logging
import os
from collections import OrderedDict

import torch

import imix.utils_imix.distributed_info as comm
from imix.utils_imix.file_io import PathManager
from .evaluator import DatasetEvaluator


class VQAEvaluator(DatasetEvaluator):
    """Evaluate object proposal, instance detection/segmentation, keypoint
    detection outputs using COCO's metrics and APIs."""

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device('cpu')
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ('classification', )
        # if cfg.MODEL.MASK_ON:
        #     tasks = tasks + ("segm", )
        # if cfg.MODEL.KEYPOINT_ON:
        #     tasks = tasks + ("keypoints", )
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a VQA model (e.g., MCAN).
                It is a list of dict. Each dict corresponds to an image and a question ,that
                contains keys like "feature", "input_ids", "input_mask", "answers_scores".
            outputs: the outputs of a VQA model. It is a list of dicts with key
                "scores" that contains :class:`scores`.
        """

        # TODO(jinliang) prediction = {image_id=xxx,question=xxx,pred_score=xxx}
        # for input, output in zip(inputs, outputs):
        #     prediction = {
        #         "input_ids": input["input_ids"],
        #         "answers_scores": input["answers_scores"]
        #     }
        #
        #     # TODO this is ugly
        #     if "scores" in output:
        #         scores = output["scores"].to(self._cpu_device)
        #         prediction["scores"] = _get_accuracy(scores)
        #
        #     self._predictions.append(prediction)
        # inputs = list2dict(inputs)  #TODO(jinliang)

        for idx in range(outputs['scores'].shape[0]):
            prediction = dict()
            prediction['question_id'] = inputs['question_id'][idx]
            prediction['answers_scores'] = inputs['answers_scores'][idx]
            score = outputs['scores'][idx].to(self._cpu_device)
            prediction['scores'] = _get_accuracy(score.view(-1, score.shape[0]))
            self._predictions.append(prediction)

    def process_submit_result(self, inputs, outpus):
        prediction = dict()
        score, label = outpus['scores'].max(1)
        for qid, l in zip(inputs['question_id'].detach().numpy(), label.cpu().detach().numpy()):
            prediction = dict()
            prediction['question_id'] = int(qid)
            prediction['answer'] = inputs['quesid2ans'][l][0]  # TODO(jinliang):two
            self._predictions.append(prediction)

    def save_submit_result(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning('[VQAEvaluator] Did not receive valid predictions.')
            return {}

        if os.path.exists(self._output_dir) is False:
            os.mkdir(self._output_dir)
        file = os.path.join(self._output_dir, 'submit_result.json')
        with open(file, 'w') as f:
            json.dump(predictions, f, indent=True)
        self._logger.info('submit file:{}  smaple_nums:{}'.format(file, len(predictions)))

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning('[VQAEvaluator] Did not receive valid predictions.')
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, 'instances_predictions.pth')
            with PathManager.open(file_path, 'wb') as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if 'classification' in self._tasks:
            self._eval_predictions(set(self._tasks), predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """Evaluate predictions on the given tasks.

        Fill self._results with the metrics of the tasks.
        """
        self._logger.info('Preparing results for COCO format ...')
        # vqa_results = list(
        #     itertools.chain([x["scores"] for x in predictions]))
        # vqa_gt = list(
        #     itertools.chain([x["answers_scores"] for x in predictions]))
        pred_length = len(predictions)
        results_size = (pred_length, predictions[0]['scores'].shape[0])
        vqa_results = torch.zeros(results_size, dtype=torch.int64)

        vqa_gt_size = (pred_length, predictions[0]['answers_scores'].shape[0])
        vqa_gt = torch.zeros(vqa_gt_size, dtype=torch.float32)

        for idx in range(pred_length):
            vqa_results[idx] = predictions[idx]['scores']
            vqa_gt[idx] = predictions[idx]['answers_scores']

        # if self._output_dir:
        #     file_path = os.path.join(self._output_dir,
        #                              "VQA_classification_results.json")
        #     self._logger.info("Saving results to {}".format(file_path))
        #     with PathManager.open(file_path, "w") as f:
        #         f.write(json.dumps(vqa_results))
        #         f.flush()

        # if not self._do_evaluation:
        #     self._logger.info("Annotations are not available for evaluation.")
        #     return

        self._logger.info('Evaluating predictions ...')
        for task in sorted(tasks):
            if task == 'classification':
                res = _evaluate_predictions_on_vqa(vqa_gt, vqa_results)
            self._results[task] = res


def _get_accuracy(output):
    output = _masked_unk_softmax(output, 1, 0)
    output = output.argmax(dim=1)  # argmax
    return output


def _masked_unk_softmax(x, dim, mask_idx):
    x1 = torch.nn.functional.softmax(x, dim=dim)
    x1[:, mask_idx] = 0
    x1_sum = torch.sum(x1, dim=1, keepdim=True)
    y = x1 / x1_sum
    return y


def _evaluate_predictions_on_vqa(vqa_gt, vqa_results):
    """Evaluate the coco results using COCOEval API."""
    assert len(vqa_results) > 0

    one_hots = vqa_gt.new_zeros(*vqa_gt.size())
    one_hots.scatter_(1, vqa_results.view(-1, 1), 1)
    scores = one_hots * vqa_gt
    vqa_accuracy = torch.sum(scores) / vqa_gt.size(0)
    logging.info('vqa_accuracy:{}'.format(vqa_accuracy))

    return vqa_accuracy
