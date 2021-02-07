# from .classifier import ClassifierLayer
# from .decoder import VISDIALPRINCIPLES_HEAD
#
# __all__ = ['ClassifierLayer', 'VISDIALPRINCIPLES_HEAD']

from .classifier_mix import ClassifierHead, BertClassifierHead, MLPClassiferHead, LogitClassifierHead, \
    LCGNClassiferHead, TripleLinearHead, WeightNormClassifierHead, R2CHead
from .decoder_mix import VisualDialogueHead, DiscQtDecoderHead, DiscByRoundDecoderHead, LanguageDecoderHead, \
    LanguageDecoder

__all__ = [
    'ClassifierHead', 'BertClassifierHead', 'MLPClassiferHead',
    'LogitClassifierHead', 'LCGNClassiferHead', 'TripleLinearHead',
    'VisualDialogueHead', 'DiscQtDecoderHead', 'DiscByRoundDecoderHead',
    'WeightNormClassifierHead', 'LanguageDecoderHead', 'LanguageDecoder',
    'R2CHead'
]
