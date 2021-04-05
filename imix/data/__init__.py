#__all__ = ["MetadataCatelog",]

from .loaders.vqa_loader import VQADATASET
from .loaders.gqa_loader import GQADATASET
from .loaders.vizwiz_loader import VizWizDATASET
from .loaders.clevr_loader import ClevrDATASET
from .loaders.textvqa_loader import TEXTVQADATASET
from .loaders.vcr_loader import VCRDATASET
from .loaders.stvqa_loader import STVQADATASET
from .loaders.ocrvqa_loader import OCRVQADATASET
# from .loaders.referit_loader import ReferitDATASET
from .loaders.refcoco_loader import RefCOCODATASET
from .loaders.refcocop_loader import RefCOCOpDATASET
from .loaders.refcocog_loader import RefCOCOgDATASET
from .loaders.refclef_loader import RefClefDATASET
from .loaders.hatefulmemes_loader import HatefulMemesDATASET
from .loaders.visualentailment_loader import VisualEntailmentDATASET

from .loaders.lxmertpretrain_loader import LXMERTPreTrainDATASET

from .builder import build_imix_test_loader, build_imix_train_loader

__all__ = [
    'VQADATASET', 'GQADATASET', 'VizWizDATASET', 'ClevrDATASET', 'TEXTVQADATASET', 'STVQADATASET', 'OCRVQADATASET',
    'VCRDATASET', 'RefCOCODATASET', 'RefCOCOpDATASET', 'RefCOCOgDATASET', 'RefClefDATASET', 'HatefulMemesDATASET',
    'VisualEntailmentDATASET', 'LXMERTPreTrainDATASET', 'build_imix_train_loader', 'build_imix_test_loader'
]
