import logging
from abc import ABCMeta, abstractmethod

import torch

from .evaluator_imix import DATASET_CONVERTER


class BaseDatasetConverter(metaclass=ABCMeta):
    CONVERTER_TO_FUNC = {'evaluator': 'evaluation', 'submitter': 'submit', 'predictor': 'predict'}
    logger = logging.getLogger(__name__)

    def __init__(self, post_process_type: str):
        self._post_process_type = post_process_type

    def convert(self, batch_data, model_outputs, *args, **kwargs):
        try:
            run_func = getattr(self, self.CONVERTER_TO_FUNC[self.post_process_type])
            return run_func(batch_data, model_outputs, *args, **kwargs)
        except KeyError:
            msg = f'The expected type are {self.CONVERTER_TO_FUNC.keys()},but got type is {self.post_process_type}'
            self.logger.info(msg)
            raise KeyError
        except Exception as e:
            self.logger.info(e)
            raise e

    @abstractmethod
    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        pass

    @abstractmethod
    def submit(self, batch_data, model_outputs, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, batch_data, model_outputs, *args, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        return 'base_dataset_converter'

    @property
    def post_process_type(self):
        return self._post_process_type

    @staticmethod
    def list_to_tensor(list_data: list) -> torch.tensor:
        # tensor_size = (len(list_data), list_data[0].shape[1])
        if len(list_data[0].shape) == 0:
            tensor_size = (len(list_data), 1)
        elif len(list_data[0].shape) == 1:
            tensor_size = (len(list_data), list_data[0].shape[0])
        else:
            tensor_size = (len(list_data), list_data[0].shape[1])
        tensor_dtype = list_data[0].dtype
        tensor_data = torch.zeros(size=tensor_size, dtype=tensor_dtype)
        for idx, data in enumerate(list_data):
            tensor_data[idx] = data

        return tensor_data

    @abstractmethod
    # def data_pre_process(self, model_outputs, labels, *args, **kwargs):
    def data_pre_process(self, model_outputs, *args, **kwargs):
        pass


@DATASET_CONVERTER.register_module()
class VQADatasetConverter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)

    def __str__(self):
        return 'vqa_dataset_converter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        from imix.models.vqa_models.mcan_mix import list2dict
        from imix.engine.organizer import is_by_iter
        if is_by_iter():
            batch_data = list2dict(batch_data)

        labels = list(batch_data['answers_scores'].split(1))
        q_ids, scores = batch_data['question_id'].split(1), model_outputs['scores'].to('cpu').split(1)
        predictions = list({'question_id': q_id, 'scores': score} for q_id, score in zip(q_ids, scores))
        # predictions, labels = self.data_pre_process(predictions, labels, *args,
        #                                             **kwargs)
        predictions = self.data_pre_process(predictions, *args, **kwargs)
        return predictions, labels

    def submit(self, batch_data, model_outputs, *args, **kwargs):
        # scores, labels = model_outputs['scores'].max(1)
        # q_ids = batch_data['question_id'].detach().numpy()
        # labels = labels.cpu().detach().numpy()
        # q2a = batch_data['quesid2ans']
        # predictions = list({
        # 	                   'question_id': int(qid),
        # 	                   'answer': q2a[l][0]
        #                    } for qid, l in zip(q_ids, labels))
        from imix.models.vqa_models.mcan_mix import list2dict
        from imix.engine.organizer import is_by_iter
        if is_by_iter():
            batch_data = list2dict(batch_data)
        q_ids, scores = batch_data['question_id'].split(1), model_outputs['scores'].to('cpu').split(1)
        predictions = list({'question_id': q_id, 'scores': score} for q_id, score in zip(q_ids, scores))
        predictions = self.data_pre_process(predictions, *args, **kwargs)
        # by yinyin q_ids should be question str;q2a should be the answer str
        q2a = batch_data['quesid2ans']
        predictions = list({
            'questionid': str(qid),
            'prediction': str(m[0][l])
        } for qid, l, m in zip(q_ids, predictions, q2a))
        return predictions

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        q_ids = batch_data['question_id'].detach().numpy()
        scores = model_outputs['scores'].cpu().detach().numpy()
        predictions = list({'question_id': int(qid), 'scores': s} for qid, s in zip(q_ids, scores))
        return predictions

    def data_pre_process(self, model_outputs, *args, **kwargs):
        # labels = self.list_to_tensor(labels)
        scores_list = list(model_output['scores'] for model_output in model_outputs)
        scores_tensor = self.list_to_tensor(scores_list)
        predictions = self._get_maxindex(scores_tensor)
        return predictions

    @staticmethod
    def _get_maxindex(output):
        output = VQADatasetConverter._masked_unk_softmax(output, 1, 0)
        output = output.argmax(dim=1)  # argmax
        return output

    @staticmethod
    def _masked_unk_softmax(x, dim, mask_idx):
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y


@DATASET_CONVERTER.register_module()
class VisDialDatasetConverter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)
        self._rank_list = []
        # self._rank_list_rnd = []
        self.num_rounds = None
        self.gt_option_inds = []
        self.gt_relevance = []
        self.gt_relevance_round_id = []
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

    def __str__(self):
        return 'visdial_datasetconverter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        # labels = list(model_outputs['target'].split(1))
        # predictions = list(model_outputs['scores'].split(1))

        # nsp_probs = F.softmax(nsp_scores, dim=1)
        # assert nsp_probs.shape[-1] == 2
        # output.append(nsp_probs[:, 0])
        self.gt_option_inds = batch_data['gt_option_inds']
        self.gt_relevance = batch_data['gt_relevance']
        self.gt_relevance_round_id = batch_data['round_id'].squeeze(1)
        predictions = self.data_pre_process(model_outputs, *args, **kwargs)
        labels = self.gt_option_inds
        predictions = self.dict_to_list(predictions)
        return predictions, labels

    def dict_to_list(self, dict_data):
        list_data = []
        keys = list(dict_data.keys())
        length = len(dict_data[keys[0]])

        for l in range(0, length):
            d = {}
            for k in keys:
                d[k] = 1 if k == 'ndcg_d' else dict_data[k][l]
            list_data.append(d)

        return list_data

    def sparse_metrics_observe(self, predicted_scores: torch.Tensor):
        target_ranks = self.gt_option_inds
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, num_rounds, num_options)
        predicted_ranks = self.scores_to_ranks(predicted_scores)
        batch_size, num_rounds, num_options = predicted_ranks.size()
        self.num_rounds = num_rounds
        # collapse batch dimension
        predicted_ranks = predicted_ranks.view(batch_size * num_rounds, num_options)

        # shape: (batch_size * num_rounds, )
        target_ranks = target_ranks.view(batch_size * num_rounds).long()

        # shape: (batch_size * num_rounds, )
        predicted_gt_ranks = predicted_ranks[torch.arange(batch_size * num_rounds), target_ranks]
        predicted_gt_ranks = predicted_gt_ranks.view(batch_size, num_rounds)
        predicted_gt_ranks = list(predicted_gt_ranks.cpu().numpy())

        return predicted_gt_ranks

    # self._rank_list.extend(list(predicted_gt_ranks.cpu().numpy()))

    # predicted_gt_ranks_rnd = predicted_gt_ranks.view(batch_size, num_rounds)
    # #  predicted gt ranks
    # self._rank_list_rnd.append(predicted_gt_ranks_rnd.cpu().numpy())

    def ndcg_observe(self, predicted_scores: torch.Tensor):
        """Observe model output scores and target ground truth relevance and
        accumulate NDCG metric.

        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense
            annotations are available for 1 randomly picked round out of 10.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth
            relevance of each answer option for a particular round.
        """
        target_relevance = self.gt_relevance
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, 1, num_options)
        predicted_scores = predicted_scores.unsqueeze(1)
        predicted_ranks = self.scores_to_ranks(predicted_scores)

        # shape: (batch_size, num_options)
        predicted_ranks = predicted_ranks.squeeze(dim=-2)
        batch_size, num_options = predicted_ranks.size()

        k = torch.sum(target_relevance != 0, dim=-1)

        # shape: (batch_size, num_options)
        _, rankings = torch.sort(predicted_ranks, dim=-1)
        # Sort relevance in descending order so highest relevance gets top rnk.
        _, best_rankings = torch.sort(target_relevance, dim=-1, descending=True)

        # shape: (batch_size, )
        batch_ndcg = []
        for batch_index in range(batch_size):
            num_relevant = k[batch_index]
            dcg = self._dcg(
                rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            best_dcg = self._dcg(
                best_rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            batch_ndcg.append(dcg / best_dcg)
        self._ndcg_denominator = batch_size
        # self._ndcg_numerator = sum(batch_ndcg)
        self._ndcg_numerator = batch_ndcg

        return self._ndcg_denominator, self._ndcg_numerator

    def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = torch.log2(torch.arange(len(rankings)).float() + 2)
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def scores_to_ranks(self, scores: torch.Tensor):
        """Convert model output scores into ranks."""
        batch_size, num_rounds, num_options = scores.size()
        scores = scores.view(-1, num_options)

        # sort in descending order - largest score gets highest rank
        sorted_ranks, ranked_idx = scores.sort(1, descending=True)

        # i-th position in ranked_idx specifies which score shall take this
        # position but we want i-th position to have rank of score at that
        # position, do this conversion
        ranks = ranked_idx.clone().fill_(0)
        for i in range(ranked_idx.size(0)):
            for j in range(num_options):
                ranks[i][ranked_idx[i][j]] = j
        # convert from 0-99 ranks to 1-100 ranks
        ranks += 1
        ranks = ranks.view(batch_size, num_rounds, num_options)
        return ranks

    def submit(self, batch_data, model_outputs, *args, **kwargs):
        scores, labels = model_outputs['scores'].max(1)
        q_ids = batch_data['question_id'].detach().numpy()
        labels = labels.cpu().detach().numpy()
        q2a = batch_data['quesid2ans']
        predictions = list({'question_id': int(qid), 'answer': q2a[l][0]} for qid, l in zip(q_ids, labels))
        return predictions

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        q_ids = batch_data['question_id'].detach().numpy()
        scores = model_outputs['scores'].cpu().detach().numpy()
        predictions = list({'question_id': int(qid), 'scores': s} for qid, s in zip(q_ids, scores))
        return predictions

    def data_pre_process(self, model_outputs, *args, **kwargs):
        output = model_outputs['nsp_scores']
        _rank_list = self.sparse_metrics_observe(output)
        output = output[torch.arange(output.size(0)), self.gt_relevance_round_id - 1, :]
        _ndcg_denominator, _ndcg_numerator = self.ndcg_observe(output)
        predictions = {'sparse_r': _rank_list, 'ndcg_d': _ndcg_denominator, 'ndcg_n': _ndcg_numerator}

        return predictions


@DATASET_CONVERTER.register_module()
class VCRDatasetConverter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)

    def __str__(self):
        return 'vcr_dataset_converter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        # from imix.models.vqa_models.mcan_mix import list2dict
        # from imix.engine.organizer import is_by_iter
        # if is_by_iter():
        #   batch_data = list2dict(batch_data)

        # labels = list(batch_data['answers_scores'].split(1))
        labels = list(model_outputs['target'].split(1))
        # q_ids, scores = batch_data['question_id'].split(
        #     1), model_outputs['scores'].to('cpu').split(1)
        # predictions = list({
        #     'question_id': q_id,
        #     'scores': score
        # } for q_id, score in zip(q_ids, scores))
        predictions = list(model_outputs['scores'].split(1))
        # predictions, labels = self.data_pre_process(predictions, labels, *args,
        #                                             **kwargs)
        predictions = self.data_pre_process(predictions, *args, **kwargs)

        return predictions, labels

    def submit(self, batch_data, model_outputs, *args, **kwargs):
        scores, labels = model_outputs['scores'].max(1)
        q_ids = batch_data['question_id'].detach().numpy()
        labels = labels.cpu().detach().numpy()
        q2a = batch_data['quesid2ans']
        predictions = list({'question_id': int(qid), 'answer': q2a[l][0]} for qid, l in zip(q_ids, labels))
        return predictions

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        q_ids = batch_data['question_id'].detach().numpy()
        scores = model_outputs['scores'].cpu().detach().numpy()
        predictions = list({'question_id': int(qid), 'scores': s} for qid, s in zip(q_ids, scores))
        return predictions

    def data_pre_process(self, model_outputs, *args, **kwargs):
        # labels = self.list_to_tensor(labels)
        # scores_list = list(model_output['scores'] for model_output in model_outputs)
        scores_list = list(model_output for model_output in model_outputs)
        scores_tensor = self.list_to_tensor(scores_list)
        predictions = VCRDatasetConverter._get_accuracy(scores_tensor)
        return predictions

    @staticmethod
    def _get_accuracy(output):
        output = VCRDatasetConverter._masked_unk_softmax(output, 1, 0)
        output = output.argmax(dim=1)  # argmax
        return output

    @staticmethod
    def _masked_unk_softmax(x, dim, mask_idx):
        # x1 = torch.nn.functional.softmax(x, dim=dim)
        x1 = torch.nn.functional.log_softmax(x, dim=dim)
        # x1[:, mask_idx] = 0
        # x1_sum = torch.sum(x1, dim=1, keepdim=True)
        # y = x1 / x1_sum
        y = x1
        return y


@DATASET_CONVERTER.register_module()
class CaptionBleu4Converter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)
        self.caption_processor = None
        # self.caption_processor = registry.get("coco_caption_processor")

    def __str__(self):
        return 'CaptionBleu4Converter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        references = []
        hypotheses = []

        # References
        targets = batch_data.answers
        for j, _ in enumerate(targets):
            img_captions = [self.caption_processor(c)['tokens'] for c in targets[j].tolist()]
            references.append(img_captions)

        # Hypotheses
        if 'captions' in model_outputs:
            scores = model_outputs['captions']
        else:
            scores = torch.max(model_outputs['scores'], dim=-1)[1]
        scores = scores.tolist()
        predictions = []
        for j, _ in enumerate(scores):
            caption = self.caption_processor(scores[j])['tokens']
            predictions.append(caption)
        hypotheses.extend(predictions)

        assert len(references) == len(hypotheses)

        return hypotheses, references

    def submit(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        pass
