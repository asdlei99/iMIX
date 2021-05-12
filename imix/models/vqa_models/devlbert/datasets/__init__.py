from .concept_cap_dataset import ConceptCapLoaderTrain, ConceptCapLoaderVal, ConceptCapLoaderRetrieval
from .vqa_dataset import VQAClassificationDataset
from .refer_expression_dataset import ReferExpressionDataset
from .retreival_dataset import RetreivalDataset, RetreivalDatasetVal
from .vcr_dataset import VCRDataset

# from .flickr_retreival_dataset import FlickrRetreivalDatasetTrain, FlickrRetreivalDatasetVal

__all__ = [
    'VQAClassificationDataset',
    'ConceptCapLoaderTrain',
    'ConceptCapLoaderVal',
    'ReferExpressionDataset',
    'RetreivalDataset',
    'RetreivalDatasetVal',
    'VCRDataset',
    'ConceptCapLoaderRetrieval',
]

DatasetMapTrain = {
    'VQA': VQAClassificationDataset,
    'VCR_Q-A': VCRDataset,
    'VCR_QA-R': VCRDataset,
    'RetrievalFlickr30k': RetreivalDataset,
    'refcoco+': ReferExpressionDataset,
}

DatasetMapEval = {
    'VQA': VQAClassificationDataset,
    'VCR_Q-A': VCRDataset,
    'VCR_QA-R': VCRDataset,
    'RetrievalFlickr30k': RetreivalDataset,
    'refcoco+': ReferExpressionDataset,
}
