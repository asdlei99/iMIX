import mix.utils.comm as comm
from .base_hook import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class EvaluateHook(HookBase):
    """Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last
    iteration.
    """

    def __init__(self, eval_period, eval_function):
        """
        Args:
            eval_period (int): the period to run `eval_function`.
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), 'Eval function must return a dict. Got {} instead.'.format(
                results)

            # flattened_results = flatten_results_dict(results)
            flattened_results = 111  #TODO(jinliang)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception:
                    raise ValueError(
                        '[EvalHook] eval_function should return a nested dict of float. '
                        "Got '{}: {}' instead.".format(k, v))
            self.trainer.log_buffer.put_scalars(
                **flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_eval()

    def after_train(self):
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func
