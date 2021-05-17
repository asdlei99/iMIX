data_root = '/home/datasets/mix_data/vilbert/datasets/'

task_ids = '1'  # '1-2-3...' training task separate by -

TASKS = dict(
    TASK1=dict(
        name='VQA',
        loss='BCEWithLogitLoss',
        process='normal',
        type='VL-classifier',
        loss_scale=1,
        dataroot=data_root + 'VQA/',
        features_h5path1=data_root + 'coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=23,
        max_region_num=101,
        train_split='trainval',
        val_split='minval',
        num_labels=3129,
        lr=0.00004,
        per_gpu_train_batch_size=128,  # 128
        per_gpu_eval_batch_size=1024,  # 1024
        num_warmup_steps=8472,  # warmup_proportion=0.1
        num_training_steps=84720,  # ceil(totoal 443753 / batch size 32/ GPUS ) * epoch size
    ),
    TASK2=dict(
        name='GenomeQA',
        type='VL-classifier',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'visual_genome/',
        features_h5path1=data_root + 'visual_genome/vg_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=26,
        max_region_num=101,
        train_split='train',
        val_split='val',
        num_labels=3129,
        lr=0.00004,
        per_gpu_train_batch_size=128,
        per_gpu_eval_batch_size=1024,
        num_warmup_steps=20224,
        num_training_steps=202240,
    ),
    TASK3=dict(
        name='VisualDialog',
        type='VL-logit',
        loss='CrossEntropyLoss',
        process='dialog',
        loss_scale=1,
        dataroot=data_root + 'visual_dialog/',
        features_h5path1=data_root + 'coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath=data_root + 'visual_dialog/visdial_1.0_train.json',
        val_annotations_jsonpath=data_root + 'visual_dialog/visdial_1.0_val.json',
        max_seq_length=16,
        max_region_num=101,
        train_split='train',
        val_split='val',
        num_labels=1,  # TODO need verified
        lr=0.00004,
        per_gpu_train_batch_size=64,
        per_gpu_eval_batch_size=64,
    ),
    TASK4=dict(
        name='Visual7w',
        type='V-logit-mc',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'visual7w/',
        features_h5path1=data_root + 'visual7w/visual7w_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'visual7w/visual7w_gt_resnext152_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=20,
        max_region_num=200,
        train_split='train',
        val_split='val',
        num_labels=1,
        lr=0.00002,
        per_gpu_train_batch_size=64,  # 256
        per_gpu_eval_batch_size=64,  # 256
        num_warmup_steps=734,
        num_training_steps=7340,
    ),
    TASK5=dict(
        name='VCR_Q-A',
        type='VL-logit',
        loss='CrossEntropyLoss',
        process='expand',
        loss_scale=1,
        dataroot=data_root + 'CVR/',
        features_h5path1=data_root + 'VCR/VCR_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'VCR/VCR_gt_resnext152_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath=data_root + 'VCR/train.jsonl',
        val_annotations_jsonpath=data_root + 'VCR/val.jsonl',
        max_seq_length=60,
        max_region_num=101,
        train_split='train',
        val_split='val',
        num_labels=1,  # TODO need verified
        lr=0.00002,
        per_gpu_train_batch_size=64,
        per_gpu_eval_batch_size=64,
    ),
    TASK6=dict(
        name='VCR_QA-R',
        type='VL-logit',
        loss='CrossEntropyLoss',
        process='expand',
        loss_scale=1,
        dataroot=data_root + 'CVR/',
        features_h5path1=data_root + 'VCR/VCR_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'VCR/VCR_gt_resnext152_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath=data_root + 'VCR/train.jsonl',
        val_annotations_jsonpath=data_root + 'VCR/val.jsonl',
        max_seq_length=80,
        max_region_num=101,
        train_split='train',
        val_split='val',
        num_labels=1,  # TODO need verified
        lr=0.00002,
        per_gpu_train_batch_size=64,
        per_gpu_eval_batch_size=64,
    ),
    TASK7=dict(
        name='RetrievalCOCO',
        type='VL-logit',
        loss='CrossEntropyLoss',
        process='retrieval',
        loss_scale=1,
        dataroot=data_root + 'cocoRetreival/',
        features_h5path1=data_root + 'coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath=data_root + 'cocoRetreival/all_data_final_train_2014.jsonline',
        val_annotations_jsonpath=data_root + 'cocoRetreival/all_data_final_test_set0_2014.jsonline',
        max_seq_length=30,
        max_region_num=101,
        train_split='train',
        val_split='val',
        num_labels=1,
        lr=0.00002,
        per_gpu_train_batch_size=32,  # 128,
        per_gpu_eval_batch_size=32,  # 128,
        num_warmup_steps=7620,
        num_training_steps=76200,
    ),
    TASK8=dict(
        name='RetrievalFlickr30k',
        type='VL-logit',
        loss='CrossEntropyLoss',
        process='retrieval',
        loss_scale=1,
        dataroot=data_root + 'flickr30k/',
        features_h5path1=data_root + 'flickr30k/flickr30k_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath=data_root + 'flickr30k/all_data_final_train_2014.jsonline',
        val_annotations_jsonpath=data_root + 'flickr30k/all_data_final_test_set0_2014.jsonline',
        max_seq_length=30,
        max_region_num=101,
        train_split='train',
        val_split='val',
        num_labels=1,
        lr=0.00002,
        per_gpu_train_batch_size=32,  # 128,
        per_gpu_eval_batch_size=32,  # 128,
        num_warmup_steps=2196,
        num_training_steps=21960,
    ),
    TASK9=dict(
        name='refcoco',
        type='V-logit',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'refcoco/',
        features_h5path1=data_root + 'refcoco/refcoco_unc/refcoco_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'refcoco/refcoco_unc/refcoco_gt_resnext152_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=20,
        max_region_num=101,
        train_split='train',
        val_split='val',
        num_labels=1,
        lr=0.00002,
        per_gpu_train_batch_size=256,
        per_gpu_eval_batch_size=256,
        num_warmup_steps=752,
        num_training_steps=7520,
    ),
    TASK10=dict(
        name='refcoco+',
        type='V-logit',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'refcoco/',
        features_h5path1=data_root + 'refcoco/refcoco+_unc/refcoco+_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'refcoco/refcoco+_unc/refcoco+_gt_resnext152_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=20,
        max_region_num=101,
        train_split='train',
        val_split='val',
        num_labels=1,
        lr=0.00002,
        per_gpu_train_batch_size=256,
        per_gpu_eval_batch_size=1024,
        num_warmup_steps=750,
        num_training_steps=7500,
    ),
    TASK11=dict(
        name='refcocog',
        type='V-logit',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'refcoco/',
        features_h5path1=data_root + 'refcoco/refcocog_umd/refcocog_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'refcoco/refcocog_umd/refcocog_gt_resnext152_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=20,
        max_region_num=101,
        train_split='train',
        val_split='val',
        num_labels=1,
        lr=0.00002,
        per_gpu_train_batch_size=64,  # 256
        per_gpu_eval_batch_size=64,  # 256
        num_warmup_steps=512,
        num_training_steps=5120,
    ),
    TASK12=dict(
        name='NLVR2',
        type='VL-binary-classifier',
        loss='BCEWithLogitLoss',
        process='nlvr',
        loss_scale=1,
        dataroot=data_root + 'nlvr2/',
        features_h5path1=data_root + 'nlvr2/nlvr2_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=40,
        max_region_num=101,
        train_split='train',
        val_split='dev',
        num_labels=1,  # TODO need verified
        lr=0.00002,
        per_gpu_train_batch_size=64,
        per_gpu_eval_batch_size=512,
    ),
    TASK13=dict(
        name='VisualEntailment',
        type='VL-tri-classifier',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'visual_entailment/',
        features_h5path1=data_root + 'flickr30k/flickr30k_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=56,
        max_region_num=101,
        train_split='train',
        val_split='dev',
        num_labels=3,
        lr=0.00002,
        per_gpu_train_batch_size=128,  # 256,
        per_gpu_eval_batch_size=512,  # 1024,
        num_warmup_steps=4004,
        num_training_steps=40040,
    ),
    TASK14=dict(
        name='GuessWhat',
        type='VL-tri-classifier',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'guesswhat/',
        features_h5path1=data_root + 'coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=25,
        max_region_num=101,
        train_split='train',
        val_split='valid',
        num_labels=3,
        lr=0.00004,
        per_gpu_train_batch_size=256,  # 256,
        per_gpu_eval_batch_size=1024,  # 1024,
        num_warmup_steps=4530,
        num_training_steps=45300,
    ),
    TASK15=dict(
        name='GQA',
        type='VL-classifier-GQA',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'gqa/',
        features_h5path1=data_root + 'gqa/gqa_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=26,
        max_region_num=101,
        train_split='trainval',
        val_split='minval',
        num_labels=1533,
        lr=0.00004,
        per_gpu_train_batch_size=128,
        per_gpu_eval_batch_size=1024,
        num_warmup_steps=15046,
        num_training_steps=150460,
    ),
    TASK16=dict(
        name='Foil',
        type='VL-binary-classifier',
        loss='CrossEntropyLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'Foil/',
        features_h5path1=data_root + 'coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath=data_root + 'Foil/foilv1.0_train_2017.json',
        val_annotations_jsonpath=data_root + 'Foil/foilv1.0_test_2017.json',
        max_seq_length=20,
        max_region_num=101,
        train_split='train',
        val_split='val',
        num_labels=1,  # TODO need verified
        lr=0.00004,
        per_gpu_train_batch_size=256,
        per_gpu_eval_batch_size=1024,
    ),
    TASK17=dict(
        name='GuessWhatPointing',
        type='V-logit-mc',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'guesswhat/',
        features_h5path1=data_root + 'coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'guesswhat/guesswhat_gt_resnext152_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=256,
        max_region_num=306,
        train_split='train',
        val_split='valid',
        num_labels=1,
        lr=0.00002,
        per_gpu_train_batch_size=16,  # 64,
        per_gpu_eval_batch_size=16,  # 64,
        num_warmup_steps=3138,
        num_training_steps=31380,
    ),
    TASK18=dict(
        name='FlickrGrounding',
        type='V-logit',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        dataroot=data_root + 'flickr30k/',
        features_h5path1=data_root + 'flickr30k/flickr30k_resnext152_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'flickr30k/flickr30k_gt_resnext152_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=24,
        max_region_num=200,
        train_split='train',
        val_split='val',
        num_labels=1,  # TODO need verified
        lr=0.000002,
        per_gpu_train_batch_size=256,
        per_gpu_eval_batch_size=256,
    ),
)
