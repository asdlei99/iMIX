dataset_type = 'VQADATASET'
data_root = '~/.cache/torch/mmf/'
feature_path = 'data/datasets/vqa2/grid_features/features/'
annotation_path = 'data/datasets/vqa2/grid_features/annotations/'
feature_default_path = 'data/datasets/vqa2/defaults/features/'
annotation_default_path = 'data/datasets/vqa2/defaults/annotations/'
vocab_path = 'data/datasets/vqa2/defaults/extras/vocabs/'

# train_datasets = ["train", "val", "visualgenome"]
# train_datasets = ["train"]
# test_datasets = ["oneval"]
# test_datasets = ["test"]

train_datasets = ['train']
test_datasets = ['minival']

vqa_reader_train_cfg = dict(
    type='VQAReader',
    card='default',
    mmf_features=dict(
        train=data_root + feature_default_path + 'trainval2014.lmdb',
        val=data_root + feature_default_path + 'trainval2014.lmdb',
        test=data_root + feature_default_path + 'test2015.lmdb',
        minival=data_root + feature_default_path + 'trainval2014.lmdb',
        train_coco10pc=data_root + feature_default_path + 'trainval2014.lmdb',
        train_coco50pc=data_root + feature_default_path + 'trainval2014.lmdb',
        valminusminival=data_root + feature_default_path + 'trainval2014.lmdb',
    ),
    mmf_annotations=dict(
        train=data_root + annotation_default_path + 'imdb_train2014.npy',
        val=data_root + annotation_default_path + 'imdb_val2014.npy',
        test=data_root + annotation_default_path + 'imdb_test2015.npy',
        minival=data_root + annotation_default_path + 'imdb_minival2014.npy',
        train_coco10pc=data_root + annotation_default_path +
        'imdb_train2014_len_coco_10_pc.npy',
        train_coco50pc=data_root + annotation_default_path +
        'imdb_train2014_len_coco_50_pc.npy',
        valminusminival=data_root + annotation_default_path +
        'imdb_valminusminival2014.npy',
    ),
    datasets=train_datasets  # used datasets
)

vqa_reader_test_cfg = dict(
    type='VQAReader',
    card='default',
    mmf_features=dict(
        train=data_root + feature_default_path + 'trainval2014.lmdb',
        val=data_root + feature_default_path + 'trainval2014.lmdb',
        test=data_root + feature_default_path + 'test2015.lmdb',
        minival=data_root + feature_default_path + 'trainval2014.lmdb',
        train_coco10pc=data_root + feature_default_path + 'trainval2014.lmdb',
        train_coco50pc=data_root + feature_default_path + 'trainval2014.lmdb',
        valminusminival=data_root + feature_default_path + 'trainval2014.lmdb',
    ),
    mmf_annotations=dict(
        train=data_root + annotation_default_path + 'imdb_train2014.npy',
        val=data_root + annotation_default_path + 'imdb_val2014.npy',
        test=data_root + annotation_default_path + 'imdb_test2015.npy',
        minival=data_root + annotation_default_path + 'imdb_minival2014.npy',
        train_coco10pc=data_root + annotation_default_path +
        'imdb_train2014_len_coco_10_pc.npy',
        train_coco50pc=data_root + annotation_default_path +
        'imdb_train2014_len_coco_50_pc.npy',
        valminusminival=data_root + annotation_default_path +
        'imdb_valminusminival2014.npy',
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
    workers_per_gpu=1,
    sampler_name='TrainingSampler',
    data=dict(
        type=dataset_type,
        vqa_reader=vqa_reader_train_cfg,
        vqa_info_cpler=vqa_info_cpler_cfg,
        limit_nums=800))

# evaluation = dict(metric=["bbox", "segm"]) TODO(jinliang) imix-evaluation
test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    sampler_name='TestingSampler',
    # metric="",
    data=dict(
        type=dataset_type,
        vqa_reader=vqa_reader_test_cfg,
        vqa_info_cpler=vqa_info_cpler_cfg),
    eval_period=5000)  # eval_period set to 0 to disable

evaluator_type = 'VQA'  # TODO(jinliang)
