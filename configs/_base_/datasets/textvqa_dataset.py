dataset_type = 'TEXTVQADATASET'
data_root = '/home/datasets/mix_data/iMIX/'
feature_path = 'data/datasets/textvqa/defaults/features/open_images/'
ocr_feature_path = 'data/datasets/textvqa/defaults/ocr/'
annotation_path = 'data/datasets/textvqa/defaults/annotations/'
vocab_path = 'data/datasets/textvqa/defaults/extras/vocabs/'

train_datasets = ['train_en']
test_datasets = ['test_en']

textvqa_reader_train_cfg = dict(
    type='TEXTVQAREADER',
    card='default',
    mix_features=dict(
        train_en=data_root + feature_path + 'detectron.lmdb',
        train_ml=data_root + feature_path + 'detectron.lmdb',
        val_en=data_root + feature_path + 'detectron.lmdb',
        val_ml=data_root + feature_path + 'detectron.lmdb',
        test_en=data_root + feature_path + 'detectron.lmdb',
        test_ml=data_root + feature_path + 'detectron.lmdb',
    ),
    mix_global_features=dict(
        train_en=data_root + feature_path + 'resnet152.lmdb',
        train_ml=data_root + feature_path + 'resnet152.lmdb',
        val_en=data_root + feature_path + 'resnet152.lmdb',
        val_ml=data_root + feature_path + 'resnet152.lmdb',
        test_en=data_root + feature_path + 'resnet152.lmdb',
        test_ml=data_root + feature_path + 'resnet152.lmdb',
    ),
    mix_ocr_features=dict(
        train_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        train_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
        val_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        val_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
        test_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        test_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
    ),
    mix_annotations=dict(
        train_en=data_root + annotation_path + 'imdb_train_ocr_en.npy',
        train_ml=data_root + annotation_path + 'imdb_train_ocr_ml.npy',
        val_en=data_root + annotation_path + 'imdb_val_ocr_en.npy',
        val_ml=data_root + annotation_path + 'imdb_val_ocr_ml.npy',
        test_en=data_root + annotation_path + 'imdb_test_ocr_en.npy',
        test_ml=data_root + annotation_path + 'imdb_test_ocr_ml.npy',
    ),
    datasets=train_datasets,
    is_global=True)

textvqa_reader_test_cfg = dict(
    type='TEXTVQAREADER',
    card='default',
    mix_features=dict(
        train_en=data_root + feature_path + 'detectron.lmdb',
        train_ml=data_root + feature_path + 'detectron.lmdb',
        val_en=data_root + feature_path + 'detectron.lmdb',
        val_ml=data_root + feature_path + 'detectron.lmdb',
        test_en=data_root + feature_path + 'detectron.lmdb',
        test_ml=data_root + feature_path + 'detectron.lmdb',
    ),
    mix_global_features=dict(
        train_en=data_root + feature_path + 'resnet152.lmdb',
        train_ml=data_root + feature_path + 'resnet152.lmdb',
        val_en=data_root + feature_path + 'resnet152.lmdb',
        val_ml=data_root + feature_path + 'resnet152.lmdb',
        test_en=data_root + feature_path + 'resnet152.lmdb',
        test_ml=data_root + feature_path + 'resnet152.lmdb',
    ),
    mix_ocr_features=dict(
        train_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        train_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
        val_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        val_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
        test_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        test_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
    ),
    mix_annotations=dict(
        train_en=data_root + annotation_path + 'imdb_train_ocr_en.npy',
        train_ml=data_root + annotation_path + 'imdb_train_ocr_ml.npy',
        val_en=data_root + annotation_path + 'imdb_val_ocr_en.npy',
        val_ml=data_root + annotation_path + 'imdb_val_ocr_ml.npy',
        test_en=data_root + annotation_path + 'imdb_test_ocr_en.npy',
        test_ml=data_root + annotation_path + 'imdb_test_ocr_ml.npy',
    ),
    datasets=train_datasets,
    is_global=True)

textvqa_info_cpler_cfg = dict(
    type='TEXTVQAInfoCpler',
    glove_weights=dict(
        glove6b50d=data_root + 'glove/glove.6B.50d.txt.pt',
        glove6b100d=data_root + 'glove/glove.6B.100d.txt.pt',
        glove6b200d=data_root + 'glove/glove.6B.200d.txt.pt',
        glove6b300d=data_root + 'glove/glove.6B.300d.txt.pt',
    ),
    fasttext_weights=dict(
        wiki300d1m=data_root + 'fasttext/wiki-news-300d-1M.vec',
        wiki300d1msub=data_root + 'fasttext/wiki-news-300d-1M-subword.vec',
        wiki_bin=data_root + 'fasttext/wiki.en.bin',
    ),
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
    mix_vocab=dict(
        answers_empty=data_root + vocab_path + 'answers_empty.txt',
        answers_textvqa_8k=data_root + vocab_path + 'answers_textvqa_8k.txt',
        answers_textvqa_more_than_1=data_root + vocab_path + 'answers_textvqa_more_than_1.txt',
        fixed_answer_vocab_textvqa_5k=data_root + vocab_path + 'fixed_answer_vocab_textvqa_5k.txt',
        vocabulary_100k=data_root + vocab_path + 'vocabulary_100k.txt',
    ),
    max_seg_lenth=14,  # 20
    max_ocr_lenth=100,
    max_txt_lenth=50,
    max_copy_steps=12,
    word_mask_ratio=0.0,
    vocab_name='vocabulary_100k',
    vocab_answer_name='answers_textvqa_8k',
    glove_name='glove6b300d',
    fasttext_name='wiki_bin',
    if_bert=True,
)

train_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=textvqa_reader_train_cfg, info_cpler=textvqa_info_cpler_cfg, limit_nums=None))

test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=textvqa_reader_test_cfg, info_cpler=textvqa_info_cpler_cfg),
)
