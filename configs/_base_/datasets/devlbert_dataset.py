dataset_type = 'DeVLBertVQADATASET'

test_datasets = ['minval']

vqa_reader_train_cfg = dict(
    type='ImageFeaturesH5Reader',
    features_path='/home/datasets/mix_data/DeVLBert/coco/coco_trainval_resnet101_faster_rcnn_genome.lmdb')

vqa_reader_train_gt_cfg = dict(
    type='ImageFeaturesH5Reader',
    features_path='/home/datasets/mix_data/DeVLBert/coco/coco_trainval_resnet101_faster_rcnn_genome.lmdb')

vqa_reader_test_cfg = dict(
    type='ImageFeaturesH5Reader',
    features_path='/home/datasets/mix_data/DeVLBert/coco/coco_test_resnet101_faster_rcnn_genome.lmdb')

vqa_reader_test_gt_cfg = dict(
    type='ImageFeaturesH5Reader',
    features_path='/home/datasets/mix_data/DeVLBert/coco/coco_test_resnet101_faster_rcnn_genome.lmdb')

train_data = dict(
    samples_per_gpu=32,  # 16
    workers_per_gpu=1,
    sampler_name='TrainingSampler',
    data=dict(
        type=dataset_type,
        task='VQA',
        dataroot='/home/datasets/mix_data/vilbert/datasets/VQA',
        annotations_jsonpath='',
        split='trainval',
        image_features_reader=vqa_reader_train_cfg,
        gt_image_features_reader=vqa_reader_train_gt_cfg,
        tokenizer='/home/datasets/VQA/bert/bert-base-uncased',
        padding_index=0,
        max_seq_length=16,
        max_region_num=100))

# just the same as train
test_data = dict(
    samples_per_gpu=128,  # 16
    workers_per_gpu=1,
    sampler_name='TestingSampler',
    data=dict(
        type=dataset_type,
        task='VQA',
        dataroot='/home/datasets/mix_data/vilbert/datasets/VQA',
        annotations_jsonpath='',
        split='minval',
        image_features_reader=vqa_reader_test_cfg,
        gt_image_features_reader=vqa_reader_test_gt_cfg,
        tokenizer='/home/datasets/VQA/bert/bert-base-uncased',
        padding_index=0,
        max_seq_length=16,
        max_region_num=100),
    eval_period=5000,
    datasets=test_datasets)

# evaluator_type = 'VQA'  # TODO(jinliang)
post_processor = dict(
    type='Evaluator', metrics=[dict(type='VQAAccuracyMetric')], dataset_converters=[dict(type='VQADatasetConverter')])
