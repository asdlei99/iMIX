"""
author: lxc
created time: 2021/1/11
"""
from .refcoco2_infocpler import RefCOCOInfoCpler


class RefClefInfoCpler(RefCOCOInfoCpler):

    def __init__(self, cfg):
        super().__init__(cfg)
