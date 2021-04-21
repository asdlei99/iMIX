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

work_dir = './work_dirs'  # the dir to save logs and models

# load_from = '/home/datasets/mix_data/model/visdial_model_imix/vqa_weights.pth'
# load_from = '/home/jinliang/iMIX/imix/work_dirs/epoch18_model.pth'

# resume_from = None
workflow = [('train', 1)]
seed = 13
CUDNN_BENCHMARK = False
model_device = 'cuda'
find_unused_parameters = True
