import enum


@enum.unique
class PriorityStatus(enum.Enum):
  HIGHEST = 0
  HIGHER = 10
  HIGH = 20
  NORMAL = 30
  LOW = 40
  LOWER = 50
  LOWEST = 60


class HookBase:
  """HookBase是所有hook的基类，在EngineBase类中进行注册.

  每个Hook实现一下6中方法，按照一下的方式进行调用：
  ::
      hook.before_train()
      for iter in range(start_iter,max_iter):
          hook.before_train_iter()
          train.run_train_iter()
          hook.after_train_iter()
      hook.after_epoch()
      hook.after_train()
  Notes:
      1. 在hook方法中，用户可以通过self.trainer获取更多的信息
  """

  def before_train(self):
    pass

  def after_train(self):
    pass

  def before_train_iter(self):
    pass

  def after_train_iter(self):
    pass

  def before_train_epoch(self):
    pass

  def after_train_epoch(self):
    pass

  def before_forward(self):
    pass

  def after_forward(self):
    pass

  @property
  def level(self):
    return self._level if hasattr(self, '_level') else PriorityStatus.NORMAL
