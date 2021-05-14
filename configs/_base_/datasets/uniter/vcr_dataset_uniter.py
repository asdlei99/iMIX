dataset_root_dir = '/home/datasets/UNITER/vcr/'

train_datasets = ['train']
test_datasets = ['minival']
vcr_cfg = dict(
    train_txt_dbs=[dataset_root_dir + 'vcr_train.db/'],
    train_img_dbs=[dataset_root_dir + 'vcr_gt_train/', dataset_root_dir + 'vcr_train/'],
    val_txt_db=dataset_root_dir + 'vcr_val.db/',
    val_img_db=[dataset_root_dir + 'vcr_gt_val/', dataset_root_dir + '/img/vcr_val/'],
    checkpoint=dataset_root_dir + 'uniter-base-vcr_2nd_stage.pt',
    max_txt_len=220,
    conf_th=0.2,
    max_bb=100,
    min_bb=10,
    num_bb=36,
    train_batch_size=4000,
    val_batch_size=10,
    gradient_accumulation_steps=5,
    learning_rate=6e-05,
    lr_mul=1.0,
    valid_steps=1000,
    num_train_steps=8000,
    optim='UNITERadamw',
    betas=[0.9, 0.98],
    dropout=0.1,
    weight_decay=0.01,
    grad_norm=2.0,
    warmup_steps=800,
    seed=42,
    fp16=True,
    n_workers=4,
    pin_mem=True)
dataset_type = 'VcrDataset'
dataset_val_type = dataset_type
train_data = dict(
    # samples_per_gpu=nlvr_cfg['train_batch_size'],  # 16
    samples_per_gpu=vcr_cfg['train_batch_size'],
    workers_per_gpu=0,
    pin_mem=True,
    batch_sampler='TokenBucketSampler',
    data=dict(type=dataset_type, datacfg=vcr_cfg, train_or_val=True))

# evaluation = dict(metric=["bbox", "segm"]) TODO(jinliang) imix-evaluation
test_data = dict(
    samples_per_gpu=vcr_cfg['val_batch_size'],
    workers_per_gpu=0,
    pin_mem=True,
    # metric="",
    batch_sampler='TokenBucketSampler',
    data=dict(type=dataset_val_type, datacfg=vcr_cfg, train_or_val=False),
    eval_period=1000)  # eval_period set to 0 to disable

# evaluator_type = 'VQA'  # TODO(jinliang)
post_processor = dict(
    type='Evaluator', metrics=[dict(type='VQAAccuracyMetric')], dataset_converters=[dict(type='VQADatasetConverter')])
