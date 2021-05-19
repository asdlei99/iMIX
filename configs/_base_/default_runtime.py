# eval_iter_period = 4000
# checkpoint_config = dict(iter_period=eval_iter_period)
log_config = dict(period=5)  # PeriodicLogger parameter
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'

work_dir = './work_dirs'  # the dir to save logs and models

# load_from = '/home/datasets/mix_data/model/visdial_model_imix/vqa_weights.pth'
# load_from = '/home/jinliang/iMIX/imix/work_dirs/epoch18_model.pth'

# resume_from = '/home/jinliang/iMIX/imix/tools/work_dirs/11/OSCAR_OSCAR_VQADataset_epoch14_model.pth'
workflow = [('train', 1)]
seed = 13
CUDNN_BENCHMARK = False
model_device = 'cuda'
find_unused_parameters = True
