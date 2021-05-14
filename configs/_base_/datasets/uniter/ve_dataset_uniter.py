dataset_type = 'VeDataset'
data_root = '/home/datasets/UNITER/ve/'
# train_datasets = ["train", "val", "visualgenome"]
# train_datasets = ["train"]
# test_datasets = ["oneval"]
# test_datasets = ["test"]

train_datasets = ['train']
test_datasets = ['minival']

ve_cfg = dict(
    train_txt_db=data_root + 've_train.db/',
    train_img_db=data_root + 'flickr30k/',
    val_txt_db=data_root + 've_dev.db/',
    val_img_db=data_root + 'flickr30k/',
    test_txt_db=data_root + 've_test.db/',
    test_img_db=data_root + 'flickr30k/',
    compressed_db=False,
    checkpoint=data_root + '../uniter-base.pt',
    max_txt_len=60,
    conf_th=0.2,
    max_bb=100,
    min_bb=10,
    num_bb=36,
    train_batch_size=4096,
    val_batch_size=4096,
    gradient_accumulation_steps=4,
    learning_rate=8e-05,
    valid_steps=500,
    num_train_steps=4000,
    optim='UNITERadamw',
    betas=[0.9, 0.98],
    dropout=0.1,
    weight_decay=0.01,
    grad_norm=2.0,
    warmup_steps=400,
    seed=42,
    fp16=True,
    n_workers=4,
    pin_mem=False)

train_data = dict(
    samples_per_gpu=ve_cfg['train_batch_size'],  # 16
    workers_per_gpu=0,
    batch_sampler='TokenBucketSampler',
    data=dict(type=dataset_type, datacfg=ve_cfg, train_or_val=True),
)

test_data = dict(
    samples_per_gpu=ve_cfg['val_batch_size'],
    workers_per_gpu=0,
    batch_sampler='TokenBucketSampler',
    data=dict(type=dataset_type, datacfg=ve_cfg, train_or_val=False),
    eval_period=5000)  # eval_period set to 0 to disable

# evaluator_type = 'VQA'  # TODO(jinliang)
post_processor = dict(
    type='Evaluator', metrics=[dict(type='VQAAccuracyMetric')], dataset_converters=[dict(type='VQADatasetConverter')])
