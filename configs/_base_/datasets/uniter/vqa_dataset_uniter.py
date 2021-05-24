dataset_type = 'UNITER_VqaDataset'
data_root = '/home/datasets/mix_data/UNITER/VQA/'

train_datasets = ['train']
test_datasets = ['minival']  # name not in use, but have defined one to run

vqa_cfg = dict(
    train_txt_dbs=[
        data_root + 'vqa_train.db',
        data_root + 'vqa_trainval.db',
        data_root + 'vqa_vg.db',
    ],
    train_img_dbs=[
        data_root + 'coco_train2014/',
        data_root + 'coco_val2014',
        data_root + 'vg/',
    ],
    val_txt_db=data_root + 'vqa_devval.db',
    val_img_db=data_root + 'coco_val2014/',
    ans2label_file=data_root + 'ans2label.json',
    max_txt_len=60,
    conf_th=0.2,
    max_bb=100,
    min_bb=10,
    num_bb=36,
    train_batch_size=20480,  # 5120,
    val_batch_size=40960,  # 10240,
    # gradient_accumulation_steps=5,
    # learning_rate=8e-05,
    # lr_mul=10.0,
    # valid_steps=500,
    # num_train_steps=6000,
    # optim='UNITERadamw',
    # betas=[0.9, 0.98],
    # dropout=0.1,
    # weight_decay=0.01,
    # grad_norm=2.0,
    # warmup_steps=600,
    # seed=42,
    # fp16=True,
    # n_workers=0,
    # pin_mem=True,
)

BUCKET_SIZE = 8192

train_data = dict(
    samples_per_gpu=vqa_cfg['train_batch_size'],
    workers_per_gpu=0,
    pin_memory=True,
    batch_sampler=dict(
        type='TokenBucketSampler',
        bucket_size=BUCKET_SIZE,
        batch_size=vqa_cfg['train_batch_size'],
        drop_last=True,
        size_multiple=8,
    ),
    data=dict(
        type=dataset_type,
        datacfg=vqa_cfg,
        train_or_val=True,
    ),
)

test_data = dict(
    samples_per_gpu=vqa_cfg['val_batch_size'],
    workers_per_gpu=0,
    batch_sampler=dict(
        type='TokenBucketSampler',
        bucket_size=BUCKET_SIZE,
        batch_size=vqa_cfg['val_batch_size'],
        drop_last=False,
        size_multiple=8,
    ),
    pin_memory=True,
    data=dict(
        type=dataset_type,
        datacfg=vqa_cfg,
        train_or_val=False,
    ),
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='UNITER_AccuracyMetric')],
    dataset_converters=[dict(type='UNITER_DatasetConverter')],
)
