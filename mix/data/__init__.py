#__all__ = ["MetadataCatelog",]

from .loaders.vqa_loader import VQADATASET
from .loaders.gqa_loader import GQADATASET
from .loaders.textvqa_loader import TEXTVQADATASET
from .loaders.vcr_loader import VCRDATASET
from .loaders.refcoco_loader import RefCOCODATASET

from .builder import build_mix_test_loader, build_mix_train_loader

__all__ = [
    'VQADATASET', 'GQADATASET', 'TEXTVQADATASET', 'VCRDATASET',
    'RefCOCODATASET', 'build_mix_train_loader', 'build_mix_test_loader'
]
