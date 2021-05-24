dataset_type = 'LXMERTPreTrainDATASET'
vg_feat_root = '/home/datasets/GQA_2/vg_gqa_imgfeat/'
coco_feat_root = '/home/datasets/VQA/mscoco_imgfeat/'
annotation_root = '/home/datasets/VQA/pretrain_json/'

feat_splits = ['coco_train', 'coco_val', 'vg_train']
annotation_splits = ['mscoco_train', 'mscoco_minival', 'mscoco_nominival', 'vgnococo']
is_debug = True
task_matched = True

reader_train_cfg = dict(
    type='LXMERTPreTrainReader',
    lxmert_feat=dict(
        coco_train=coco_feat_root + 'train2014_obj36.tsv',
        coco_val=coco_feat_root + 'val2014_obj36.tsv',
        vg_train=vg_feat_root + 'vg_gqa_obj36.tsv',
    ),
    lxmert_annotation=dict(
        mscoco_train=annotation_root + 'mscoco_train.json',
        mscoco_minival=annotation_root + 'mscoco_minival.json',
        mscoco_nominival=annotation_root + 'mscoco_nominival.json',
        vgnococo=annotation_root + 'vgnococo.json',
    ),
    feat_splits=['coco_train', 'coco_val', 'vg_train'],
    annotation_splits=['mscoco_train', 'mscoco_nominival', 'vgnococo'],
    vocab_ans_path=annotation_root + 'all_ans.json',
    training=True,
    is_debug=is_debug,
    task_matched=task_matched,
    num_random_feats=5,
)

reader_test_cfg = dict(
    type='LXMERTPreTrainReader',
    lxmert_feat=dict(
        coco_train=coco_feat_root + 'train2014_obj36.tsv',
        coco_val=coco_feat_root + 'val2014_obj36.tsv',
        vg_train=vg_feat_root + 'vg_gqa_obj36.tsv',
    ),
    lxmert_annotation=dict(
        mscoco_train=annotation_root + 'mscoco_train.json',
        mscoco_minival=annotation_root + 'mscoco_minival.json',
        mscoco_nominival=annotation_root + 'mscoco_nominival.json',
        vgnococo=annotation_root + 'vgnococo.json',
    ),
    feat_splits=['coco_val'],
    annotation_splits=['mscoco_minival'],
    vocab_ans_path=annotation_root + 'all_ans.json',
    training=False,
    is_debug=is_debug,
    task_matched=task_matched,
    num_random_feats=3,
)

info_cpler_cfg = dict(
    type='LXMERTPreTrainInfoCpler',
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
    max_seq_length=20,
    word_mask_ratio=0.15,
    roi_mask_ratio=0.15,
    if_bert=True,
)

train_data = dict(
    samples_per_gpu=4,  # 16
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=reader_train_cfg, info_cpler=info_cpler_cfg, limit_nums=400))

test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=reader_test_cfg, info_cpler=info_cpler_cfg),
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='LXMERTPreTrainAccuracyMetric')],
    dataset_converters=[dict(type='LXMERTPreTrainDatasetConverter')])
