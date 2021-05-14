dataset_root_dir = '/home/datasets/UNITER/nlvr2/'
# train_datasets = ["train", "val", "visualgenome"]
# train_datasets = ["train"]
# test_datasets = ["oneval"]
# test_datasets = ["test"]

train_datasets = ['train']
test_datasets = ['minival']
nlvr_cfg = dict(
    train_txt_db=dataset_root_dir + 'nlvr2_train.db',
    train_img_db=dataset_root_dir + 'nlvr2_train/',
    val_txt_db=dataset_root_dir + 'nlvr2_dev.db',
    val_img_db=dataset_root_dir + 'nlvr2_dev/',
    test_txt_db=dataset_root_dir + 'nlvr2_test1.db',
    test_img_db=dataset_root_dir + 'nlvr2_test/',
    checkpoint=dataset_root_dir + '../uniter-base.pt',
    use_img_type=True,
    max_txt_len=60,
    conf_th=0.2,
    max_bb=100,
    min_bb=10,
    num_bb=36,
    train_batch_size=10240,
    val_batch_size=10240,
    gradient_accumulation_steps=1,
    learning_rate=3e-05,
    valid_steps=500,
    num_train_steps=8000,
    optim='UNITERadamw',
    betas=[0.9, 0.98],
    dropout=0.1,
    weight_decay=0.01,
    grad_norm=2.0,
    warmup_steps=800,
    seed=77,
    fp16=True,
    n_workers=4,
)
dataset_type = 'NLVR2Dataset'
train_data = dict(
    # samples_per_gpu=nlvr_cfg['train_batch_size'],  # 16
    samples_per_gpu=nlvr_cfg['train_batch_size'],
    workers_per_gpu=0,
    pin_mem=True,
    batch_sampler='TokenBucketSampler',
    data=dict(type=dataset_type, datacfg=nlvr_cfg, train_or_val=True))

# evaluation = dict(metric=["bbox", "segm"]) TODO(jinliang) imix-evaluation
test_data = dict(
    samples_per_gpu=nlvr_cfg['val_batch_size'],
    workers_per_gpu=0,
    batch_sampler='TokenBucketSampler',
    pin_mem=True,
    # metric="",
    data=dict(type=dataset_type, datacfg=nlvr_cfg, train_or_val=False),
    eval_period=500)  # eval_period set to 0 to disable

# evaluator_type = 'VQA'  # TODO(jinliang)
post_processor = dict(
    type='Evaluator', metrics=[dict(type='VQAAccuracyMetric')], dataset_converters=[dict(type='VQADatasetConverter')])
