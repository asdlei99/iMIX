checkpoint_config = dict(period=50)
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
load_from = '/home/jinliang/epoch_9.pth'
# load_from = None
resume_from = None
workflow = [('train', 1)]
SEED = 39189013
CUDNN_BENCHMARK = False
model_device = 'cuda'
find_unused_parameters = True
