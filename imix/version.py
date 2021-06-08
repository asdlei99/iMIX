# Copyright (c) Inspur, Inc. and its affiliates. All Rights Reserved

__version__ = '1.0.0'
short_version = __version__


def parse_version_info(version_str):
    ver_info = [d for d in version_str.split('.') if d.isdigit()]
    return tuple(ver_info)


version_info = parse_version_info(__version__)
