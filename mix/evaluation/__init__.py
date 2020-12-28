# from .evaluator import DatasetEvaluator, DatasetEvaluators, inference_on_dataset, inference_context, build_submit_file, \
#     build_test_predict_result
# from .vqa_evaluation import VQAEvaluator

from .testing import print_csv_format, verify_results, flatten_results_dict

from .vqa_evaluation_mix import VQAEvaluator
from .evaluator_mix import DatasetEvaluator, DatasetEvaluators, inference_on_dataset, inference_context, \
    build_submit_file, build_test_predict_result

__all__ = [
    'DatasetEvaluator', 'DatasetEvaluators', 'inference_on_dataset',
    'inference_context', 'print_csv_format', 'verify_results',
    'flatten_results_dict', 'VQAEvaluator', 'build_submit_file',
    'build_test_predict_result'
]
