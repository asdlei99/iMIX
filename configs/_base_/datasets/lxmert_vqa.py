dataset_type = 'VQATorchDataset'
data_root = '/home/datasets/mix_data/lxmert/'
feature_path = 'mscoco_imgfeat/'
annotation_path = 'vqa/'

train_datasets = ['train', 'nominival']
test_datasets = ['minival']

vqa_reader_train_cfg = dict(
    annotations=dict(
        train=data_root + annotation_path + 'train.json',
        test=data_root + annotation_path + 'test.json',
        minival=data_root + annotation_path + 'minival.json',
        nominival=data_root + annotation_path + 'nominival.json',
    ),
    answer_2_label=data_root + annotation_path + 'trainval_ans2label.json',
    label_2_answer=data_root + annotation_path + 'trainval_label2ans.json',
    datasets=train_datasets,  # used datasets
    # topk=32,
    img_feature=dict(
        train=data_root + feature_path + 'train2014_obj36.tsv',
        valid=data_root + feature_path + 'val2014_obj36.tsv',
        minival=data_root + feature_path + 'val2014_obj36.tsv',
        nominival=data_root + feature_path + 'val2014_obj36.tsv',
        test=data_root + feature_path + 'test2015_obj36.tsv',
    ))

vqa_reader_test_cfg = dict(
    annotations=dict(
        train=data_root + annotation_path + 'train.json',
        test=data_root + annotation_path + 'test.json',
        minival=data_root + annotation_path + 'minival.json',
        nominival=data_root + annotation_path + 'nominival.json',
    ),
    answer_2_label=data_root + annotation_path + 'trainval_ans2label.json',
    label_2_answer=data_root + annotation_path + 'trainval_label2ans.json',
    # topk=32,
    datasets=test_datasets,  # used datasets
    img_feature=dict(
        train=data_root + feature_path + 'train2014_obj36.tsv',
        valid=data_root + feature_path + 'val2014_obj36.tsv',
        minival=data_root + feature_path + 'val2014_obj36.tsv',
        nominival=data_root + feature_path + 'val2014_obj36.tsv',
        test=data_root + feature_path + 'test2015_obj36.tsv',
    ))

train_data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    sampler_name='TrainingSampler',
    data=dict(
        type=dataset_type,
        reader=vqa_reader_train_cfg,
        # limit_nums=400,
    ),
    drop_last=True,
    shuffle=True,
)

test_data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    sampler_name='TestingSampler',
    data=dict(type=dataset_type, reader=vqa_reader_test_cfg),
    drop_last=False,
    shuffle=False,
    eval_period=5000,
    limit_nums = 800
)  # eval_period set to 0 to disable

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='LXMERT_VQAAccuracyMetric', cfg=vqa_reader_test_cfg, task='VQA')],
    dataset_converters=[dict(type='LXMERT_VQADatasetConverter')])
