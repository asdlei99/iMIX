"""
author: lxc
created time: 2020/8/17
"""


class BaseDataReader():

  def __init__(self):
    # load config: path, name, split ...
    pass

  def load(self):
    pass

  def __getitem__(self, item):
    pass

  def deduplist(self, l):
    return list(set(l))
