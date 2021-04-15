# __all__ = ["MetadataCatelog",]

from .builder import build_imix_test_loader, build_imix_train_loader
from .loaders.clevr_loader import ClevrDATASET
from .loaders.gqa_loader import GQADATASET
from .loaders.hatefulmemes_loader import HatefulMemesDATASET
from .loaders.lxmertpretrain_loader import LXMERTPreTrainDATASET
from .loaders.ocrvqa_loader import OCRVQADATASET
from .loaders.refclef_loader import RefClefDATASET
# from .loaders.referit_loader import ReferitDATASET
from .loaders.refcoco_loader import RefCOCODATASET
from .loaders.refcocog_loader import RefCOCOgDATASET
from .loaders.refcocop_loader import RefCOCOpDATASET
from .loaders.stvqa_loader import STVQADATASET
from .loaders.textvqa_loader import TEXTVQADATASET
from .loaders.vcr_loader import VCRDATASET
from .loaders.visualentailment_loader import VisualEntailmentDATASET

from .loaders.lxmertpretrain_loader import LXMERTPreTrainDATASET
from .loaders.visual_dialog_dataset import VisDialDataset
from .loaders.vqa_loader import VQADATASET
from .loaders.vizwiz_loader import VizWizDATASET

from .loaders.lxmert_vqa import VQATorchDataset
from .loaders.lxmert_nlvr2 import NLVR2TorchDataset
from .loaders.lxmert_gqa import GQATorchDataset
from .loaders.devlbert_vqa_loader import DeVLBertVQADATASET

__all__ = [
    'VQADATASET', 'GQADATASET', 'VizWizDATASET', 'ClevrDATASET', 'TEXTVQADATASET', 'STVQADATASET', 'OCRVQADATASET',
    'VCRDATASET', 'RefCOCODATASET', 'RefCOCOpDATASET', 'RefCOCOgDATASET', 'RefClefDATASET', 'HatefulMemesDATASET',
    'VisualEntailmentDATASET', 'LXMERTPreTrainDATASET', 'VQATorchDataset', 'NLVR2TorchDataset', 'GQATorchDataset',
    'build_imix_train_loader', 'build_imix_test_loader', 'VisDialDataset', 'DeVLBertVQADATASET'
]
