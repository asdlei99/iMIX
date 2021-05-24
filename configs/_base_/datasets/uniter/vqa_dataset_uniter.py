dataset_type = 'VqaDataset'
data_root = '/home/datasets/mix_data/mmf/'
# train_datasets = ["train", "val", "visualgenome"]
# train_datasets = ["train"]
# test_datasets = ["oneval"]
# test_datasets = ["test"]

train_datasets = ['train']
test_datasets = ['minival']

vqa_cfg = dict(
    train_txt_dbs=[
        '/home/datasets/UNITER/VQA/vqa_train.db', '/home/datasets/UNITER/VQA/vqa_trainval.db',
        '/home/datasets/UNITER/VQA/vqa_vg.db'
    ],
    train_img_dbs=[
        '/home/datasets/UNITER/VQA/coco_train2014/', '/home/datasets/UNITER/VQA/coco_val2014',
        '/home/datasets/UNITER/VQA/vg/'
    ],
    val_txt_db='/home/datasets/UNITER/VQA/vqa_devval.db',
    val_img_db='/home/datasets/UNITER/VQA/coco_val2014/',
    ans2label_file='/home/datasets/UNITER/VQA/ans2label.json',
    max_txt_len=60,
    conf_th=0.2,
    max_bb=100,
    min_bb=10,
    num_bb=36,
    train_batch_size=5120,
    val_batch_size=10240,
    gradient_accumulation_steps=5,
    learning_rate=8e-05,
    lr_mul=10.0,
    valid_steps=500,
    num_train_steps=6000,
    optim='UNITERadamw',
    betas=[0.9, 0.98],
    dropout=0.1,
    weight_decay=0.01,
    grad_norm=2.0,
    warmup_steps=600,
    seed=42,
    fp16=True,
    n_workers=0,
    pin_mem=True)

train_data = dict(
    samples_per_gpu=vqa_cfg['train_batch_size'],
    workers_per_gpu=0,
    pin_mem=True,
    batch_sampler='TokenBucketSampler',
    data=dict(type=dataset_type, datacfg=vqa_cfg, train_or_val=True))

test_data = dict(
    samples_per_gpu=vqa_cfg['val_batch_size'],
    workers_per_gpu=0,
    batch_sampler='TokenBucketSampler',
    pin_mem=True,
    data=dict(type=dataset_type, datacfg=vqa_cfg, train_or_val=False),
)

post_processor = dict(
    type='Evaluator', metrics=[dict(type='VQAAccuracyMetric')], dataset_converters=[dict(type='VQADatasetConverter')])
