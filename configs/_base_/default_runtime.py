checkpoint_config = dict(period=100)
# yapf:disable
log_config = dict(
    period=2,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs'  # the dir to save logs and models
load_from = '/home/jinliang/code/Mix/mix/work_dir/model_0000099.pth'
resume_from = None
workflow = [('train', 1)]
SEED = 100
CUDNN_BENCHMARK = False
model_device = 'cuda'
