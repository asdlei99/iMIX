import logging
from abc import ABCMeta, abstractmethod

import torch

from .evaluator_mix1 import DATASET_CONVERTER


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
            self.logger.info('The expected type are {},but got type is {}'.format(self.CONVERTER_TO_FUNC.keys(),
                                                                                  self.post_process_type))
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
