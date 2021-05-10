# TODO(jinliang):jinliang_copy
# from iopath.common.file_io import PathManager
import os
import tempfile

import torch

from imix.utils_imix.file_io import PathManager
from .base_hook import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class AutogradProfilerHook(HookBase):
    """A hook which runs `torch.autograd.profiler.profile`.

    Examples:

    .. code-block:: python

        hooks.AutogradProfiler(
             lambda trainer: trainer.iter > 10 and trainer.iter < 20, self.cfg.OUTPUT_DIR
        )

    The above example will run the profiler for iteration 10~20 and dump
    results to ``OUTPUT_DIR``. We did not profile the first few iterations
    because they are typically slower than the rest.
    The result files can be loaded in the ``chrome://tracing`` page in chrome browser.

    Note:
        When used together with NCCL on older version of GPUs,
        autograd profiler may cause deadlock because it unnecessarily allocates
        memory on every device it sees. The memory management calls, if
        interleaved with NCCL calls, lead to deadlock on GPUs that do not
        support `cudaLaunchCooperativeKernelMultiDevice`.
    """

    def __init__(self, enable_predicate, output_dir, *, use_cuda=True):
        """
        Args:
            enable_predicate (callable[trainer -> bool]): a function which takes a trainer,
                and returns whether to enable the profiler.
                It will be called once every step, and can be used to select which steps to profile.
            output_dir (str): the output directory to dump tracing files.
            use_cuda (bool): same as in `torch.autograd.profiler.profile`.
        """
        super().__init__()
        self._enable_predicate = enable_predicate
        self._use_cuda = use_cuda
        self._output_dir = output_dir

    def before_train_iter(self):
        if self._enable_predicate(self.trainer):
            self._profiler = torch.autograd.profiler.profile(use_cuda=self._use_cuda)
            self._profiler.__enter__()
        else:
            self._profiler = None

    def after_train_iter(self):
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        PathManager.mkdirs(self._output_dir)
        out_file = os.path.join(self._output_dir, 'profiler-trace-iter{}.json'.format(self.trainer.iter))
        if '://' not in out_file:
            self._profiler.export_chrome_trace(out_file)
        else:
            # Support non-posix filesystems
            with tempfile.TemporaryDirectory(prefix='imix_profiler') as d:
                tmp_file = os.path.join(d, 'tmp.json')
                self._profiler.export_chrome_trace(tmp_file)
                with open(tmp_file) as f:
                    content = f.read()
            with PathManager.open(out_file, 'w') as f:
                f.write(content)
