checkpoint_config = dict(period=5000)
# yapf:disable
log_config = dict(
    period=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/wbq/code2/imix/work_dir'  # the dir to save logs and models
#load_from = '/home/wbq/code2/imix/work_dir/model_epoch3.pth'
load_from = '/home/wbq/code2/imix/work_dir/epoch23_model.pth'
resume_from = None
workflow = [('train', 1)]
seed = 9595#13
CUDNN_BENCHMARK = False
model_device = 'cuda'
find_unused_parameters = True
