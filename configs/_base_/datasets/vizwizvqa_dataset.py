dataset_type = 'VizWizDATASET'
data_root = '/home/datasets/mix_data/mmf/'
feature_default_path = 'data/datasets/vizwiz/defaults/features/'
annotation_default_path = 'data/datasets/vizwiz/defaults/annotations/'
vocab_path = 'data/datasets/vizwiz/defaults/extras/vocabs/'

train_datasets = ['train']
test_datasets = ['val']

reader_train_cfg = dict(
    type='VizWizReader',
    card='default',
    mix_features=dict(
        train=data_root + feature_default_path + 'detectron.lmdb',
        val=data_root + feature_default_path + 'detectron.lmdb',
        test=data_root + feature_default_path + 'detectron.lmdb',
    ),
    mix_annotations=dict(
        train=data_root + annotation_default_path + 'imdb_vizwiz_train.npy',
        val=data_root + annotation_default_path + 'imdb_vizwiz_val.npy',
        test=data_root + annotation_default_path + 'imdb_vizwiz_test.npy',
    ),
    datasets=train_datasets  # used datasets
)

reader_test_cfg = dict(
    type='VizWizReader',
    card='default',
    mix_features=dict(
        train=data_root + feature_default_path + 'detectron.lmdb',
        val=data_root + feature_default_path + 'detectron.lmdb',
        test=data_root + feature_default_path + 'detectron.lmdb',
    ),
    mix_annotations=dict(
        train=data_root + annotation_default_path + 'imdb_vizwiz_train.npy',
        val=data_root + annotation_default_path + 'imdb_vizwiz_val.npy',
        test=data_root + annotation_default_path + 'imdb_vizwiz_test.npy',
    ),
    datasets=test_datasets  # used datasets
)

info_cpler_cfg = dict(
    type='VizWizInfoCpler',
    glove_weights=dict(
        glove6b50d=data_root + 'glove/glove.6B.50d.txt.pt',
        glove6b100d=data_root + 'glove/glove.6B.100d.txt.pt',
        glove6b200d=data_root + 'glove/glove.6B.200d.txt.pt',
        glove6b300d=data_root + 'glove/glove.6B.300d.txt.pt',
    ),
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
    mix_vocab=dict(
        answers_7k=data_root + vocab_path + 'answers_vizwiz_7k.txt',
        answers_large=data_root + vocab_path + 'answers_vizwiz_large',
        vocabulart_100k=data_root + vocab_path + 'vocabulary_100k.txt',
    ),
    max_seg_lenth=20,
    word_mask_ratio=0.0,
    vocab_name='vocabulart_100k',
    vocab_answer_name='answers_7k',
    glove_name='glove6b300d',
    if_bert=True,
)

train_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    sampler_name='TrainingSampler',
    data=dict(type=dataset_type, reader=reader_train_cfg, info_cpler=info_cpler_cfg, limit_nums=800))

# evaluation = dict(metric=["bbox", "segm"]) TODO(jinliang) mix-evaluation
test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    sampler_name='TestingSampler',
    # metric="",
    data=dict(type=dataset_type, reader=reader_test_cfg, info_cpler=info_cpler_cfg),
    eval_period=5000)  # eval_period set to 0 to disable

evaluator_type = 'VizWiz'  # TODO(jinliang)
