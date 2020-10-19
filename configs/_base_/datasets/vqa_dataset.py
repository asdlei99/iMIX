dataset_type = 'VQADATASET'
data_root = '/home/jinliang/.cache/torch/mmf/'
feature_path = 'data/datasets/vqa2/grid_features/features/'
annotation_path = 'data/datasets/vqa2/grid_features/annotations/'
vocab_path = 'data/datasets/vqa2/defaults/extras/vocabs/'

train_datasets = ['train', 'val', 'visualgenome']
test_datasets = ['oneval']

vqa_reader_train_cfg = dict(
    type='VQAReader',
    mmf_features=dict(
        train=data_root + feature_path + 'train2014',
        val=data_root + feature_path + 'val2014',
        test=data_root + feature_path + 'test2015',
        visualgenome=data_root + feature_path + 'visualgenome',
        oneval=data_root + feature_path + 'val2014'),
    mmf_annotations=dict(
        train=data_root + annotation_path + 'imdb_train2014.npy',
        val=data_root + annotation_path + 'imdb_val2014.npy',
        test=data_root + annotation_path + 'imdb_test2015.npy',
        visualgenome=data_root + annotation_path + 'imdb_visualgenome.npy',
        oneval=data_root + annotation_path + 'imdb_oneval2014.npy',
    ),
    datasets=train_datasets  # used datasets
)

vqa_reader_test_cfg = dict(
    type='VQAReader',
    mmf_features=dict(
        train=data_root + feature_path + 'train2014',
        val=data_root + feature_path + 'val2014',
        test=data_root + feature_path + 'test2015',
        visualgenome=data_root + feature_path + 'visualgenome',
        oneval=data_root + feature_path + 'val2014'),
    mmf_annotations=dict(
        train=data_root + annotation_path + 'imdb_train2014.npy',
        val=data_root + annotation_path + 'imdb_val2014.npy',
        test=data_root + annotation_path + 'imdb_test2015.npy',
        visualgenome=data_root + annotation_path + 'imdb_visualgenome.npy',
        oneval=data_root + annotation_path + 'imdb_oneval2014.npy',
    ),
    datasets=test_datasets  # used datasets
)

vqa_info_cpler_cfg = dict(
    type='VQAInfoCpler',
    glove_weights=data_root + 'glove.6B.300d.txt.pt',
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
    mmf_vocab=dict(
        answers_vqa=data_root + vocab_path + 'answers_vqa.txt',
        vocabulart_100k=data_root + vocab_path + 'vocabulary_100k.txt',
        vocabulary_vqa=data_root + vocab_path + 'vocabulary_vqa.txt'))

train_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    sampler_name='TrainingSampler',
    data=dict(
        type=dataset_type,
        vqa_reader=vqa_reader_train_cfg,
        vqa_info_cpler=vqa_info_cpler_cfg))

# evaluation = dict(metric=['bbox', 'segm']) TODO(jinliang) mix-evaluation
test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    sampler_name='TestingSampler',
    # metric='',
    data=dict(
        type=dataset_type,
        vqa_reader=vqa_reader_test_cfg,
        vqa_info_cpler=vqa_info_cpler_cfg),
    eval_period=100)  # eval_period set to 0 to disable

evaluator_type = 'VQA'  # TODO(jinliang)
