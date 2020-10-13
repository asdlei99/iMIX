class HookBase:
    """HookBase是所有hook的基类，在EngineBase类中进行注册.

    每个Hook实现一下6中方法，按照一下的方式进行调用：
    ::
        hook.before_train()
        for iter in range(start_iter,max_iter):
            hook.before_iter()
            train.run_iter()
            hook.after_iter()
        hook.after_epoch()
        hook.after_train()
    Notes:
        1. 在hook方法中，用户可以通过self.trainer获取更多的信息
    """

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_iter(self):
        pass

    def after_iter(self):
        pass
