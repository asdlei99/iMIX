import torch
from .evaluator_imix import METRICS
from abc import abstractmethod, ABCMeta


class BaseMetric(metaclass=ABCMeta):
    metric_name = 'base_metric'

    def evaluate(self, model_outputs, labels, *args, **kwargs):
        assert len(model_outputs) > 0
        # predictions, labels = self.data_pre_process(model_outputs, labels, *args,
        #                                             **kwargs)
        # predictions = self.model_outputs
        # labels = self.labels
        # metric_result = self.calculate(predictions, labels)
        metric_result = self.calculate(model_outputs, labels)
        return metric_result

    # @staticmethod
    # def list_to_tensor(list_data: list) -> torch.tensor:
    #   # tensor_size = (len(list_data), list_data[0].shape[1])
    #   if len(list_data[0].shape) == 1:
    #     tensor_size = (len(list_data), list_data[0].shape[0])
    #   else:
    #     tensor_size = (len(list_data), list_data[0].shape[1])
    #   tensor_dtype = list_data[0].dtype
    #   tensor_data = torch.zeros(size=tensor_size, dtype=tensor_dtype)
    #   for idx, data in enumerate(list_data):
    #     tensor_data[idx] = data
    #
    #   return tensor_data

    def __str__(self):
        return self.metric_name

    # @abstractmethod
    # def data_pre_process(self, model_outputs, labels, *args, **kwargs):
    #   pass

    @abstractmethod
    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        pass


@METRICS.register_module()
class VQAAccuracyMetric(BaseMetric):
    metric_name = 'vqa_accuracy_metric'

    def __init__(self, *args, **kwargs):
        pass

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        one_hots = labels.new_zeros(*labels.shape)
        one_hots.scatter_(1, predictions.view(-1, 1), 1)
        accuracy = torch.sum(one_hots * labels) / labels.shape[0]
        return accuracy


# def data_pre_process(self, model_outputs, labels, *args, **kwargs):
#   labels = self.list_to_tensor(labels)
#   scores_list = list(model_output['scores'] for model_output in model_outputs)
#   scores_tensor = self.list_to_tensor(scores_list)
#   predictions = VQAAccuracyMetric._get_accuracy(scores_tensor)
#   return predictions, labels

# @staticmethod
# def _get_accuracy(output):
#   output = VQAAccuracyMetric._masked_unk_softmax(output, 1, 0)
#   output = output.argmax(dim=1)  # argmax
#   return output
#
# @staticmethod
# def _masked_unk_softmax(x, dim, mask_idx):
#   x1 = torch.nn.functional.softmax(x, dim=dim)
#   x1[:, mask_idx] = 0
#   x1_sum = torch.sum(x1, dim=1, keepdim=True)
#   y = x1 / x1_sum
#   return y


@METRICS.register_module()
class VisDialMetric(BaseMetric):
    metric_name = 'visdialMetric'

    def __init__(self, *args, **kwargs):
        self.all_metrics = {}
        self.predictions = {}
        self._rank_list = []
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

    def calculate(self, predictions, labels, **kwargs):
        self.predictions = predictions
        for p in predictions:
            self._rank_list.extend(p['sparse_r'])
            self._ndcg_numerator += p['ndcg_n']
            self._ndcg_denominator += p['ndcg_d']
        self.all_metrics.update(self.sparse_retrieve())
        self.all_metrics.update(self.ndcg_retrieve())

        return self.all_metrics

    def sparse_retrieve(self):

        # _rank_list for p1 in predictions for _rank_list in p1['sparse_r']]
        # _rank_list = self.predictions['sparse_r']
        num_examples = len(self._rank_list)
        if num_examples > 0:
            # convert to numpy array for easy calculation.
            __rank_list = torch.tensor(self._rank_list).float()
            metrics = {
                'r@1': torch.mean((__rank_list <= 1).float()).item(),
                'r@5': torch.mean((__rank_list <= 5).float()).item(),
                'r@10': torch.mean((__rank_list <= 10).float()).item(),
                'mean': torch.mean(__rank_list).item(),
                'mrr': torch.mean(__rank_list.reciprocal()).item()
            }
            # # add round metrics
            # _rank_list_rnd = np.concatenate(self._rank_list_rnd)
            # _rank_list_rnd = _rank_list_rnd.astype(float)
            # r_1_rnd = np.mean(_rank_list_rnd <= 1, axis=0)
            # r_5_rnd = np.mean(_rank_list_rnd <= 5, axis=0)
            # r_10_rnd = np.mean(_rank_list_rnd <= 10, axis=0)
            # mean_rnd = np.mean(_rank_list_rnd, axis=0)
            # mrr_rnd = np.mean(np.reciprocal(_rank_list_rnd), axis=0)
            #
            # for rnd in range(1, self.num_rounds + 1):
            #     metrics["r_1" + "_round_" + str(rnd)] = r_1_rnd[rnd-1]
            #     metrics["r_5" + "_round_" + str(rnd)] = r_5_rnd[rnd-1]
            #     metrics["r_10" + "_round_" + str(rnd)] = r_10_rnd[rnd-1]
            #     metrics["mean" + "_round_" + str(rnd)] = mean_rnd[rnd-1]
            #     metrics["mrr" + "_round_" + str(rnd)] = mrr_rnd[rnd-1]
        else:
            metrics = {}

        return metrics

    def ndcg_retrieve(self):
        # _ndcg_denominator = sum(self.predictions['ndcg_d'])
        # _ndcg_numerator = sum(self.predictions['ndcg_n'])
        if self._ndcg_denominator > 0:
            metrics = {'ndcg': float(self._ndcg_numerator / self._ndcg_denominator)}
        else:
            metrics = {}

        return metrics


@METRICS.register_module()
class VCRAccuracyMetric(BaseMetric):
    metric_name = 'vcr_accuracy_metric'

    def __init__(self, *args, **kwargs):
        pass

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        # one_hots = labels.new_zeros(*labels.shape)
        # one_hots.scatter_(1, predictions.view(-1, 1), 1)
        # accuracy = torch.sum(one_hots * labels) / labels.shape[0]
        accuracy = torch.sum(torch.eq(labels, predictions.view(-1, 1))).float() / labels.shape[0]
        # predictions = predictions.view(-1, 1)
        # labels = np.array(labels)
        # predictions = np.array(predictions)
        # accuracy = len(set(labels) & set(predictions)) / labels.shape[0]
        # cal_num = 0
        # for i in range(len(labels)):
        #   if labels[i][0] == predictions.view(-1, 1)[i][0]:
        #     cal_num += 1
        # accuracy = cal_num / labels.shape[0]
        return accuracy


@METRICS.register_module()
class AccuracyMetric(BaseMetric):
    metric_name = 'accuracy_metric'

    def __init__(self, *args, **kwargs):
        pass

    def data_pre_process(self, model_outputs, labels, *args, **kwargs):
        labels_tensor = self.list_to_tensor(labels)
        scores_list = list(model_output['scores'] for model_output in model_outputs)
        scores_tensor = self.list_to_tensor(scores_list)

        predictions = torch.max(scores_tensor, 1)[1] if scores_tensor.dim() == 2 else scores_tensor
        flag = (labels_tensor.dim() == 2 and labels_tensor.size(-1) != 1)
        labels = torch.max(labels_tensor, 1)[1] if flag else labels_tensor

        return predictions, labels

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        accuracy = (labels == predictions.square()).sum().float() / len(labels)
        return accuracy


@METRICS.register_module()
class CaptionBleu4Metric(BaseMetric):
    metric_name = 'caption_bleu4_metric'

    def __init__(self, *args, **kwargs):
        import nltk.translate.bleu_score as bleu_score_func
        self._bleu_score_func = bleu_score_func
        self._caption_processor = None

    def evaluate(self, model_outputs, labels, *args, **kwargs):
        references, hypotheses = self.data_pre_process(model_outputs, labels)
        bleu4_score = self.calculate(references, hypotheses)
        return bleu4_score

    def data_pre_process(self, model_outputs, labels, *args, **kwargs):
        return labels, model_outputs

    def calculate(self, references: list, hypotheses: list, **kwargs):
        assert len(references) == len(hypotheses)
        bleu4_score = self._bleu_score_func.corpus_bleu(references, hypotheses)
        return bleu4_score


@METRICS.register_module()
class VQAEvalAIAccuracyMetric(BaseMetric):
    metric_name = 'vqa_eval_ai_accuracy_metric'

    def __init__(self, *args, **kwargs):
        self.evalai_answer_processor = None  # EvalAIAnswerProcessor()

    def data_pre_process(self, model_outputs, labels, *args, **kwargs):
        pass

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor):
        pass


@METRICS.register_module()
class RecallAtK(BaseMetric):
    metric_name = 'recall@k'

    def __init__(self, top_k: int = None, *args, **kwargs):
        self.top_k = top_k

    @property
    def top_k(self):
        return self.top_k

    @top_k.setter
    def top_k(self, top_k):
        self.top_k = top_k

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        k = kwargs['k'] if 'k' in kwargs.keys() else ValueError('kwargs have not k ')
        ranks = self.get_ranks(labels, predictions)
        recall = float(torch.sum(torch.le(ranks, k))) / ranks.size(0)
        return recall

    def get_ranks(self, labels, predictions, *args, **kwargs):

        ranks = self.score_to_ranks(predictions)
        gt_ranks = self.get_gt_ranks(ranks, labels)

        ranks = self.process_ranks(gt_ranks)
        return ranks.float()

    @staticmethod
    def score_to_ranks(scores):
        # sort in descending order - largest score gets highest rank
        sorted_ranks, ranked_idx = scores.sort(1, descending=True)

        # convert from ranked_idx to ranks
        ranks = ranked_idx.clone().fill_(0)
        for i in range(ranked_idx.size(0)):
            for j in range(100):
                ranks[i][ranked_idx[i][j]] = j
        ranks += 1
        return ranks

    @staticmethod
    def get_gt_ranks(ranks, ans_ind):
        _, ans_ind = ans_ind.max(dim=1)
        ans_ind = ans_ind.view(-1)
        gt_ranks = torch.LongTensor(ans_ind.size(0))

        for i in range(ans_ind.size(0)):
            gt_ranks[i] = int(ranks[i, ans_ind[i].long()])
        return gt_ranks

    def data_pre_process(self, model_outputs, labels, *args, **kwargs):
        predictions = self.list_to_tensor(model_outputs['scores'])
        labels = self.list_to_tensor(labels)
        return predictions, labels


@METRICS.register_module()
class RecallAt1(RecallAtK):
    metric_name = 'r@1'

    def __init__(self, *args, **kwargs):
        super().__init__(top_k=1, *args, **kwargs)


@METRICS.register_module()
class RecallAt5(RecallAtK):
    metric_name = 'r@5'

    def __init__(self, *args, **kwargs):
        super().__init__(top_k=5, *args, **kwargs)


@METRICS.register_module()
class RecallAt10(RecallAtK):
    metric_name = 'r@10'

    def __init__(self, *args, **kwargs):
        super().__init__(top_k=10, *args, **kwargs)


@METRICS.register_module()
class MeanRankMetric(RecallAtK):
    """Calculate MeanRank which specifies what was the average rank of the
    chosen candidate.

    **Key**: ``mean_rank``.
    """
    metric_name = 'mean_rank'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        ranks = self.get_ranks(labels, predictions)
        return torch.mean(ranks)


@METRICS.register_module()
class MeanReciprocalRankMetric(RecallAtK):
    metric_name = 'mean_reciprocal_rank'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        ranks = self.get_ranks(labels, predictions)
        return torch.mean(ranks.reciprocal())


@METRICS.register_module()
class TextVQAAccuracyMetric(BaseMetric):
    metric_name = 'text_vqa_accuracy_metric'

    def __init__(self, *args, **kwargs):
        pass

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        pass

    def data_pre_process(self, model_outputs, labels, *args, **kwargs):
        pass


@METRICS.register_module()
class STVQA_ANLS_Metric(TextVQAAccuracyMetric):
    metric_name = 'stvqa_anls_metric'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@METRICS.register_module()
class STVQAAccuracyMetric(TextVQAAccuracyMetric):
    metric_name = 'stvqa_accuracy_metric'

    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@METRICS.register_module()
class OCRVQAAccuracyMetric(STVQAAccuracyMetric):
    metric_name = 'ocrvqa_accuracy_metric'

    def __init__(self, *args, **kwargs):
        super().__int__(*args, **kwargs)


@METRICS.register_module()
class TextCapsBleu4Metric(TextVQAAccuracyMetric):
    metric_name = 'textcaps_bleu4_metric'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@METRICS.register_module()
class F1Metric(BaseMetric):
    metric_name = 'f1_metric'

    def __init__(self, *args, **kwargs):
        pass

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        pass

    def data_pre_process(self, model_outputs, labels, *args, **kwargs):
        pass


@METRICS.register_module()
class MacroF1Metric(F1Metric):
    metric_name = 'macro_f1_metric'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
