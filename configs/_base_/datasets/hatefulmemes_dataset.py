dataset_type = 'HatefulMemesDATASET'
data_root = '/home/datasets/mix_data/mmf/'
annotation_dir = '/home/datasets/hateful_memes/annotations/'
image_dir = '/home/datasets/hateful_memes/img/'
mix_dir = '/home/datasets/mix_data/mmf/data/datasets/hateful-memes/defaults/'

train_datasets = ['train']
test_datasets = ['dev']

reader_train_cfg = dict(
    type='HatefulMemesReader',
    card='default',
    mix_features=dict(
        train=mix_dir + 'features/' + 'detectron.lmdb',
        dev=mix_dir + 'features/' + 'detectron.lmdb',
        test=mix_dir + 'features/' + 'detectron.lmdb',
    ),
    mix_annotations=dict(
        train=annotation_dir + 'train.jsonl',
        dev=annotation_dir + 'dev.jsonl',
        test=annotation_dir + 'test.jsonl',
    ),
    image_dir=image_dir,
    img_size=256,
    datasets=train_datasets  # used datasets
)

reader_test_cfg = dict(
    type='HatefulMemesReader',
    card='default',
    mix_features=dict(
        train=mix_dir + 'features/' + 'detectron.lmdb',
        dev=mix_dir + 'features/' + 'detectron.lmdb',
        test=mix_dir + 'features/' + 'detectron.lmdb',
    ),
    mix_annotations=dict(
        train=annotation_dir + 'train.jsonl',
        dev=annotation_dir + 'dev.jsonl',
        test=annotation_dir + 'test.jsonl',
    ),
    image_dir=image_dir,
    img_size=256,
    datasets=test_datasets  # used datasets
)

info_cpler_cfg = dict(
    type='HatefulMemesInfoCpler',
    glove_weights=dict(
        glove6b50d=data_root + 'glove/glove.6B.50d.txt.pt',
        glove6b100d=data_root + 'glove/glove.6B.100d.txt.pt',
        glove6b200d=data_root + 'glove/glove.6B.200d.txt.pt',
        glove6b300d=data_root + 'glove/glove.6B.300d.txt.pt',
    ),
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
    mix_vocab=dict(
        vocabulary_100k=mix_dir + 'extras/vocabs/' + 'vocabulary_100k.txt', ),
    default_max_length=20,
    word_mask_ratio=0.0,
    vocab_name='vocabulary_100k',
    glove_name='glove6b300d',
    if_bert=True,
)

train_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    sampler_name='TrainingSampler',
    data=dict(
        type=dataset_type,
        reader=reader_train_cfg,
        info_cpler=info_cpler_cfg,
        limit_nums=800))

# evaluation = dict(metric=["bbox", "segm"]) TODO(jinliang) mix-evaluation
test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    sampler_name='TestingSampler',
    # metric="",
    data=dict(
        type=dataset_type, reader=reader_test_cfg, info_cpler=info_cpler_cfg),
    eval_period=5000)  # eval_period set to 0 to disable

evaluator_type = 'VQA'  # TODO(jinliang)
