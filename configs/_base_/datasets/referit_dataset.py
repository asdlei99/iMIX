dataset_type = 'ReferitDATASET'
data_root = '/home/datasets/REFCOCO/referit/'
split_path = 'splits/referit/'
vocab_path = 'splits/'
feature_default_path = ''
annotation_default_path = data_root
# vocab_path = 'data/datasets/gqa/defaults/extras/vocabs/'

train_datasets = ['train']
test_datasets = ['val']

reader_train_cfg = dict(
    type='ReferitReader',
    card='default',
    image_dir=data_root + 'images/',
    mask_dir=data_root + 'mask/',
    annotations=dict(
        imlist=dict(
            all=data_root + split_path + 'referit_all_imlist.txt',
            trainval=data_root + split_path + 'referit_trainval_imlist.txt',
            train=data_root + split_path + 'referit_train_imlist.txt',
            val=data_root + split_path + 'referit_val_imlist.txt',
            test=data_root + split_path + 'referit_test_imlist.txt',
        ),
        query=dict(
            all=data_root + split_path + 'referit_query_all.json',
            trainval=data_root + split_path + 'referit_query_trainval.json',
            train=data_root + split_path + 'referit_query_trainval.json',
            val=data_root + split_path + 'referit_query_trainval.json',
            test=data_root + split_path + 'referit_query_trainval.json',
        ),
    ),
    imcrop=data_root + split_path + 'referit_imcrop.json',
    imsize=data_root + split_path + 'referit_imsize.json',
    bbox=data_root + split_path + 'referit_bbox.json',
    datasets=train_datasets,  # used datasets
    is_train=True,
    img_size=256,
)

reader_test_cfg = dict(
    type='ReferitReader',
    card='default',
    image_dir=data_root + 'images/',
    mask_dir=data_root + 'mask/',
    annotations=dict(
        imlist=dict(
            all=data_root + split_path + 'referit_all_imlist.txt',
            trainval=data_root + split_path + 'referit_trainval_imlist.txt',
            train=data_root + split_path + 'referit_train_imlist.txt',
            val=data_root + split_path + 'referit_val_imlist.txt',
            test=data_root + split_path + 'referit_test_imlist.txt',
        ),
        query=dict(
            all=data_root + split_path + 'referit_query_all.json',
            trainval=data_root + split_path + 'referit_query_trainval.json',
            train=data_root + split_path + 'referit_query_train.json',
            val=data_root + split_path + 'referit_query_val.json',
            test=data_root + split_path + 'referit_query_test.json',
        ),
    ),
    imcrop=data_root + split_path + 'referit_imcrop.json',
    imsize=data_root + split_path + 'referit_imsize.json',
    bbox=data_root + split_path + 'referit_bbox.json',
    datasets=test_datasets,  # used datasets
    is_train=False,
    img_size=256,
)

info_cpler_cfg = dict(
    type='ReferitInfoCpler',
    default_max_length=20,
    glove_weights=dict(
        glove6b50d=data_root + 'glove/glove.6B.50d.txt.pt',
        glove6b100d=data_root + 'glove/glove.6B.100d.txt.pt',
        glove6b200d=data_root + 'glove/glove.6B.200d.txt.pt',
        glove6b300d=data_root + 'glove/glove.6B.300d.txt.pt',
    ),
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
    mix_vocab=dict(
        vocabulary_Gref=data_root + vocab_path + 'vocabulary_Gref.txt',
        vocabulary_referit=data_root + vocab_path + 'vocabulary_referit.txt',
    ),
    vocab_name='vocabulary_referit',
    vocab_answer_name='vocabulary_Gref',
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

# evaluation = dict(metric=["bbox", "segm"]) TODO(jinliang) imix-evaluation
test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    sampler_name='TestingSampler',
    # metric="",
    data=dict(
        type=dataset_type, reader=reader_test_cfg, info_cpler=info_cpler_cfg),
    eval_period=5000)  # eval_period set to 0 to disable

evaluator_type = 'Referit'  # TODO(jinliang)
