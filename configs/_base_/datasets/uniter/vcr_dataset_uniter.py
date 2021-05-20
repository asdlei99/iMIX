dataset_type = 'UNITER_VcrDataset'

dataset_root_dir = '/home/datasets/mix_data/UNITER/vcr/'

train_datasets = ['train']
test_datasets = ['minival']

vcr_cfg = dict(
    train_txt_dbs=[dataset_root_dir + 'vcr_train.db/'],
    train_img_dbs=[
        '{}vcr_gt_train/;{}vcr_train/'.format(dataset_root_dir, dataset_root_dir),
    ],  # two dbs concatenate one string
    val_txt_db=dataset_root_dir + 'vcr_val.db/',
    val_img_db='{}vcr_gt_val/;{}vcr_val/'.format(dataset_root_dir, dataset_root_dir),
    # checkpoint=dataset_root_dir + 'uniter-base-vcr_2nd_stage.pt',
    max_txt_len=220,
    conf_th=0.2,
    max_bb=100,
    min_bb=10,
    num_bb=36,
    train_batch_size=4000,  # 16000
    val_batch_size=10,  # 40
    # gradient_accumulation_steps=5,
    # learning_rate=6e-05,
    # lr_mul=1.0,
    # valid_steps=1000,
    # num_train_steps=8000,
    # optim='UNITERadamw',
    # betas=[0.9, 0.98],
    # dropout=0.1,
    # weight_decay=0.01,
    # grad_norm=2.0,
    # warmup_steps=800,
    # seed=42,
    # fp16=True,
    # n_workers=4,
    # pin_mem=True,
)

BUCKET_SIZE = 8192

train_data = dict(
    samples_per_gpu=vcr_cfg['train_batch_size'],
    workers_per_gpu=0,
    pin_memory=True,
    batch_sampler=dict(
        type='TokenBucketSampler',
        bucket_size=BUCKET_SIZE,
        batch_size=vcr_cfg['train_batch_size'],
        drop_last=True,
        size_multiple=8,
    ),
    data=dict(
        type=dataset_type,
        datacfg=vcr_cfg,
        train_or_val=True,
    ),
)

test_data = dict(
    samples_per_gpu=vcr_cfg['val_batch_size'],
    workers_per_gpu=0,
    pin_memory=True,
    data=dict(
        type=dataset_type,
        datacfg=vcr_cfg,
        train_or_val=False,
    ),
)

post_processor = dict(
    type='Evaluator',
    metrics=[
        dict(
            type='UNITER_VCR_AccuracyMetric',
            name='vcr q->a',
            metric_key='qa_batch_score',
        ),
        dict(
            type='UNITER_VCR_AccuracyMetric',
            name='vcr qa->r',
            metric_key='qar_batch_score',
        ),
        dict(
            type='UNITER_VCR_AccuracyMetric',
            name='vcr q->ar',
            metric_key='batch_score',
        ),
    ],
    dataset_converters=[dict(type='UNITER_VCR_DatasetConverter')],
)
