dataset_type = 'DEVLLoadDatasets'

data_root = '/home/datasets/mix_data/DeVLBert/'

task_ids = '10',
# 1,2,3,4,5-->1,5,6,14,10  to maintain consistency with vilbert
# test mode directly read this data set
test_datasets = ['val']
'''
test_datasets = ['minval']  # for TASK1
test_datasets = ['val']  # for TASK5
test_datasets = ['val']  # for TASK6
test_datasets = ['val']  # for TASK14
test_datasets = ['val']  # for TASK10
'''
limit_nums = None

vqa_reader_train_cfg = dict(
    tasks=task_ids,
    bert_model='/home/datasets/VQA/bert/bert-base-uncased',
    do_lower_case=True,
    gradient_accumulation_steps=1,
    in_memory=False,  # whether use chunck for parallel training
    clean_datasets=False,  # whether clean train sets for multitask data
    limit_nums=limit_nums,
    TASK1=dict(
        name='VQA',
        dataroot=data_root + 'vqa/',
        features_h5path1=data_root + 'vqa/coco/coco_trainval_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2='',
        annotations_jsonpath='',
        max_seq_length=16,
        max_region_num=100,
        split='trainval'),
    TASK5=dict(
        name='VCR_Q-A',
        dataroot=data_root + 'vcr/',
        features_h5path1=data_root + 'vcr/VCR_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'vcr/VCR_gt_resnet101_faster_rcnn_genome.lmdb',
        annotations_jsonpath=data_root + 'vcr/train.jsonl',
        max_seq_length=60,
        max_region_num=100,
        split='train'),
    TASK6=dict(
        name='VCR_QA-R',
        dataroot=data_root + 'vcr/',
        features_h5path1=data_root + 'vcr/VCR_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'vcr/VCR_gt_resnet101_faster_rcnn_genome.lmdb',
        annotations_jsonpath=data_root + 'vcr/train.jsonl',
        max_seq_length=80,
        max_region_num=100,
        split='train'),
    TASK14=dict(
        name='RetrievalFlickr30k',
        dataroot=data_root + 'flickr30k/',
        features_h5path1=data_root + 'flickr30k/flickr30k_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        annotations_jsonpath=data_root + 'flickr30k/all_data_final_train_2014.jsonline',
        max_seq_length=30,
        max_region_num=100,
        split='train'),
    TASK10=dict(
        name='refcoco+',
        dataroot=data_root,
        features_h5path1=data_root + 'refcoco+/refcoco+_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'refcoco+/refcoco+_gt_resnet101_faster_rcnn_genome.lmdb',
        annotations_jsonpath='',
        max_seq_length=20,
        max_region_num=100,
        split='train'),
)

vqa_reader_test_cfg = dict(
    tasks=task_ids,
    bert_model='/home/datasets/VQA/bert/bert-base-uncased',
    do_lower_case=True,
    gradient_accumulation_steps=1,
    in_memory=False,
    clean_datasets=True,
    limit_nums=limit_nums,
    TASK1=dict(
        name='VQA',
        dataroot=data_root + 'vqa/',
        features_h5path1=data_root + 'vqa/coco/coco_trainval_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2='',
        annotations_jsonpath='',
        max_seq_length=16,
        max_region_num=100,
        split='minval'),
    TASK5=dict(
        name='VCR_Q-A',
        dataroot=data_root + 'vcr/',
        features_h5path1=data_root + 'vcr/VCR_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'vcr/VCR_gt_resnet101_faster_rcnn_genome.lmdb',
        annotations_jsonpath=data_root + 'vcr/val.jsonl',
        max_seq_length=60,
        max_region_num=100,
        split='val'),
    TASK6=dict(
        name='VCR_QA-R',
        dataroot=data_root + 'CVR/',
        features_h5path1=data_root + 'vcr/VCR_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'vcr/VCR_gt_resnet101_faster_rcnn_genome.lmdb',
        annotations_jsonpath=data_root + 'vcr/val.jsonl',
        max_seq_length=80,
        max_region_num=100,
        split='val'),
    TASK14=dict(
        name='RetrievalFlickr30k',
        dataroot=data_root + 'flickr/',
        features_h5path1=data_root + 'flickr/flickr30k_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2='',
        annotations_jsonpath=data_root + 'flickr/all_data_final_test_set0_2014.jsonline',
        max_seq_length=30,
        max_region_num=100,
        split='val'),
    TASK10=dict(
        name='refcoco+',
        dataroot=data_root + 'refcoco+/',
        features_h5path1=data_root + 'refcoco+/refcoco+_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'refcoco+/refcoco+_gt_resnet101_faster_rcnn_genome.lmdb',
        annotations_jsonpath='',
        max_seq_length=20,
        max_region_num=100,
        split='val'),
)

train_data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    sampler_name='TrainingSampler',
    data=dict(
        type=dataset_type,
        reader=vqa_reader_train_cfg,
    ),
    pin_memory=True,
    sampler='RandomSampler',
)
'''
samples_per_gpu=256,    # for TASK1
samples_per_gpu=16,    # for TASK5
samples_per_gpu=16,     # for TASK6
samples_per_gpu=64,    # for TASK14
samples_per_gpu=256,     # for TASK10
'''
test_data = dict(
    samples_per_gpu=1024,
    workers_per_gpu=0,
    sampler_name='TestingSampler',
    data=dict(type=dataset_type, reader=vqa_reader_test_cfg),
    pin_memory=True,
    eval_period=0)  # eval_period set to 0 to disable
'''
samples_per_gpu=1024,    # for TASK1
samples_per_gpu=16,    # for TASK5
samples_per_gpu=16,     # for TASK6
samples_per_gpu=64,    # for TASK14
samples_per_gpu=256,     # for TASK10
'''
post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='DEVLBERT_AccuracyMetric')],
    dataset_converters=[dict(type='DEVLBERT_DatasetConverter')])
