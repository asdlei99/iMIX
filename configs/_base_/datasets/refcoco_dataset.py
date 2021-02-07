dataset_type = 'RefCOCODATASET'
data_root = '/home/datasets/REFCOCO/refcoco/'
# vocab_path = 'data/datasets/gqa/defaults/extras/vocabs/'

train_datasets = ['train']
test_datasets = ['val']

reader_train_cfg = dict(
    type='RefCOCOReader',
    card='default',
    data_type='unc',
    imgset_dir='/home/zrz/code/ReSC/data/unc',
    image_dir='/home/datasets/COCO2014/train2014',
    annotations=dict(
        instances=data_root + 'instances.json',
        refs_google=data_root + 'refs(google).p',
        refs_unc=data_root + 'refs(unc).p',
    ),
    datasets=train_datasets,  # used datasets
    augment=False,  #True
    is_train=True,
    img_size=256,
)

reader_test_cfg = dict(
    type='RefCOCOReader',
    card='default',
    annotations=dict(
        instances=data_root + 'instances.json',
        refs_google=data_root + 'refs(google).p',
        refs_unc=data_root + 'refs(unc).p',
    ),
    datasets=test_datasets,  # used datasets
    is_train=True,
    img_size=256,
)

info_cpler_cfg = dict(
    type='RefCOCOInfoCpler',
    default_max_length=20,
    glove_weights=dict(
        glove6b50d=data_root + 'glove/glove.6B.50d.txt.pt',
        glove6b100d=data_root + 'glove/glove.6B.100d.txt.pt',
        glove6b200d=data_root + 'glove/glove.6B.200d.txt.pt',
        glove6b300d=data_root + 'glove/glove.6B.300d.txt.pt',
    ),
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
)

train_data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    sampler_name='TrainingSampler',
    data=dict(
        type=dataset_type,
        reader=reader_train_cfg,
        info_cpler=info_cpler_cfg,
        limit_nums=800))

# evaluation = dict(metric=["bbox", "segm"]) TODO(jinliang) imix-evaluation
test_data = dict(
    samples_per_gpu=8,  #16
    workers_per_gpu=1,
    sampler_name='TestingSampler',
    # metric="",
    data=dict(
        type=dataset_type, reader=reader_test_cfg, info_cpler=info_cpler_cfg),
    eval_period=5000)  # eval_period set to 0 to disable

# evaluator_type = 'RefCOCO'  # TODO(jinliang)
post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='VQAAccuracyMetric')],
    dataset_converters=[dict(type='VQADatasetConverter')])
