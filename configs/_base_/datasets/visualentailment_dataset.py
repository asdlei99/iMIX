dataset_type = 'VisualEntailmentDATASET'
data_root = '/home/datasets/mix_data/iMIX/'
annotation_dir = '/home/datasets/mix_data/iMIX/data/datasets/visual_entailment/defaults/annotations/'
image_dir = ''
mix_dir = '/home/datasets/mix_data/iMIX/data/datasets/visual_entailment/defaults/'

train_datasets = ['train']
test_datasets = ['dev']

reader_train_cfg = dict(
    type='VisualEntailmentReader',
    card='default',
    mix_features=dict(
        train=mix_dir + 'features/' + 'detectron.lmdb',
        dev=mix_dir + 'features/' + 'detectron.lmdb',
        test=mix_dir + 'features/' + 'detectron.lmdb',
    ),
    mix_annotations=dict(
        train=annotation_dir + 'snli_ve_train.jsonl',
        dev=annotation_dir + 'snli_ve_dev.jsonl',
        test=annotation_dir + 'snli_ve_test.jsonl',
    ),
    image_dir=image_dir,
    img_size=256,
    datasets=train_datasets  # used datasets
)

reader_test_cfg = dict(
    type='VisualEntailmentReader',
    card='default',
    mix_features=dict(
        train=mix_dir + 'features/' + 'detectron.lmdb',
        dev=mix_dir + 'features/' + 'detectron.lmdb',
        test=mix_dir + 'features/' + 'detectron.lmdb',
    ),
    mix_annotations=dict(
        train=annotation_dir + 'snli_ve_train.jsonl',
        dev=annotation_dir + 'snli_ve_dev.jsonl',
        test=annotation_dir + 'snli_ve_test.jsonl',
    ),
    image_dir=image_dir,
    img_size=256,
    datasets=test_datasets  # used datasets
)

info_cpler_cfg = dict(
    type='VisualEntailmentInfoCpler',
    glove_weights=dict(
        glove6b50d=data_root + 'glove/glove.6B.50d.txt.pt',
        glove6b100d=data_root + 'glove/glove.6B.100d.txt.pt',
        glove6b200d=data_root + 'glove/glove.6B.200d.txt.pt',
        glove6b300d=data_root + 'glove/glove.6B.300d.txt.pt',
    ),
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
    mix_vocab=dict(vocabulary_100k=mix_dir + 'extras/vocabs/' + 'vocabulary_100k.txt', ),
    default_max_length=20,
    word_mask_ratio=0.0,
    vocab_name='vocabulary_100k',
    glove_name='glove6b300d',
    if_bert=True,
)

train_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=reader_train_cfg, info_cpler=info_cpler_cfg, limit_nums=800))

test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=reader_test_cfg, info_cpler=info_cpler_cfg),
)

evaluator_type = 'VQA'  # TODO(jinliang)
