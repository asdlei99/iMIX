import datetime
import logging
from collections import OrderedDict
from contextlib import contextmanager
import torch

# from mix.utils.comm import get_world_size, is_main_process
from mix.utils_mix.distributed_info import get_world_size, is_main_process
from mix.utils.logger import log_every_n_seconds
import pickle as pkl
from mix.utils_mix.Timer import Timer


class DatasetEvaluator:

    def reset(self):
        pass

    def eval_process(self, inputs, outputs):
        pass

    def evaluate(self):
        pass

    def submit_process(self, inputs, outputs):
        pass

    def save_submit_result(self):
        pass


class DatasetEvaluators(DatasetEvaluator):

    def __int__(self, evaluators: list):
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def eval_process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.eval_process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for key, value in result.items():
                    if key in results:
                        raise KeyError(
                            'This evaluator {} is already in results'.format(
                                key))
                    else:
                        results[key] = value
        return results

    def submit_process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.submit_process(inputs, outputs)

    def save_submit_result(self):
        for evaluator in self._evaluators:
            evaluator.save_submit_result()


def inference_on_dataset(model, data_loader, evaluator):
    """Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately. The model
    will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_imgs = len(data_loader)
    if evaluator is None:
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    logger = logging.getLogger(__name__)
    logger.info('Starting inference on {} images'.format(num_imgs))

    start_time = Timer.now()
    total_inference_time = 0
    with to_inference(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader, start=1):
            start_infer_time = Timer.now()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            single_infer_time = Timer.passed_seconds(
                start=start_infer_time, end=Timer.now())
            total_inference_time += single_infer_time
            evaluator.eval_process(inputs, outputs)
            if idx % 10 == 0 or idx == num_imgs:
                total_seconds_per_img = total_inference_time / idx
                eta_time = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (num_imgs - idx)))
                logger.info(
                    'Inference completion ratio {}/{} {:.4f}s/img, ETA:{}'.
                    format(idx, num_imgs, single_infer_time, eta_time))
    total_time = Timer.passed_seconds(start_time, Timer.now())
    world_size = get_world_size()
    infer_log_msg = 'Total inference time:{} speed:{:.6f} s/img per device, on {} devices'.format(
        total_inference_time, total_inference_time / (num_imgs * world_size),
        world_size)
    logger.info(infer_log_msg)
    total_log_msg = 'Total inference time:{} speed:{:.6f} s/img per device, on {} devices '.format(
        total_time, total_time / (num_imgs * world_size), world_size)
    logger.info(total_log_msg)

    results = evaluator.evaluate()
    return results if results is not None else {}


def build_submit_file(model, data_loader, evaluator):
    """Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately. The model
    will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_imgs = len(data_loader)
    if evaluator is None:
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    logger = logging.getLogger(__name__)
    logger.info('Starting inference on {} images'.format(num_imgs))

    start_time = Timer.now()
    total_inference_time = 0
    with to_inference(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader, start=1):
            start_infer_time = Timer.now()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            single_infer_time = Timer.passed_seconds(
                start=start_infer_time, end=Timer.now())
            total_inference_time += single_infer_time
            evaluator.submit_process(inputs, outputs)
            if idx % 10 == 0 or idx == num_imgs:
                total_seconds_per_img = total_inference_time / idx
                eta_time = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (num_imgs - idx)))
                logger.info(
                    'Inference completion ratio {}/{} {:.4f}s/img, ETA:{}'.
                    format(idx, num_imgs, single_infer_time, eta_time))
    total_time = Timer.passed_seconds(start_time, Timer.now())
    world_size = get_world_size()
    infer_log_msg = 'Total inference time:{} speed:{:.6f} s/img per device, on {} devices'.format(
        total_inference_time, total_inference_time / (num_imgs * world_size),
        world_size)
    logger.info(infer_log_msg)
    total_log_msg = 'Total inference time:{} speed:{:.6f} s/img per device, on {} devices '.format(
        total_time, total_time / (num_imgs * world_size), world_size)
    logger.info(total_log_msg)

    evaluator.save_submit_result()


def build_test_predict_result(model, data_loader):
    """Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately. The model
    will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_imgs = len(data_loader)
    logger = logging.getLogger(__name__)
    logger.info('Start inference on {} images'.format(num_imgs))
    test_predict = dict()
    with to_inference(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader, start=1):
            logger.info('idx:{}/{}'.format(idx, num_imgs))
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for qid, pred in zip(inputs['question_id'].detach().numpy(),
                                 outputs['scores'].cpu().detach().numpy()):
                test_predict['question_id'] = qid
                test_predict['scores'] = pred

    if is_main_process():
        logger.info('vqa data size:{}'.format(len(test_predict)))
        save_path = '/home/jinliang/code/Mix/mix/work_dir/inference/vqa_test_predict.pkl'
        with open(save_path, 'wb') as f:
            pkl.dump(test_predict, f)
        logger.info('vqa_test_predict save path:{}'.format(save_path))


@contextmanager
def to_inference(model):
    old_mode = model.training
    model.eval()
    yield
    model.train(old_mode)


inference_context = to_inference
