"""
author: lxc
created time: 2021/1/11
"""
from ..utils.stream import ItemFeature
from .refcoco2_infocpler import RefCOCOInfoCpler
from torchvision import transforms as T


class RefClefInfoCpler(RefCOCOInfoCpler):

    def __init__(self, cfg):
        super().__init__(cfg)
