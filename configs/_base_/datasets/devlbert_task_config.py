data_root = '/home/datasets/mix_data/DeVLBert/'

task_ids = '3'  # '1-2-3...' training task separate by -

TASKS = dict(
    TASK1=dict(
        name='VQA',
        loss='BCEWithLogitLoss',
        process='normal',
        type='VL-classifier',
        loss_scale=1,
        task_id=1,
        dataroot=data_root + 'vqa/',
        features_h5path1=data_root + 'vqa/coco/coco_trainval_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=16,
        max_region_num=100,
        train_split='trainval',
        val_split='minval',
        num_labels=3129,
        lr=0.00004,
        per_gpu_train_batch_size=32,  # 256,
        per_gpu_eval_batch_size=128,  # 1024,
        iters_in_epoch=16,  # 2560,
        num_warmup_steps=32,  # 5120,  # warmup_proportion=0.1
        num_training_steps=320,  # 51200,  # ceil(totoal 443753 / batch size 32/ GPUS ) * epoch size
        num_epoch=20,
    ),
    TASK2=dict(
        name='VCR_Q-A',
        type='VL-logit',
        loss='CrossEntropyLoss',
        process='expand',
        loss_scale=1,
        task_id=6,
        dataroot=data_root + 'vcr/',
        features_h5path1=data_root + 'vcr/VCR_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'vcr/VCR_gt_resnet101_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath=data_root + 'vcr/train.jsonl',
        val_annotations_jsonpath=data_root + 'vcr/val.jsonl',
        max_seq_length=60,
        max_region_num=100,
        train_split='train',
        val_split='val',
        num_labels=1,
        lr=0.00002,
        per_gpu_train_batch_size=16,  # 16,
        per_gpu_eval_batch_size=16,  # 16,
        iters_in_epoch=13308,  # 13308,
        num_warmup_steps=13308,  # 26616,  # 13308,
        num_training_steps=266160,  # 26616,  # 133080,
        num_epoch=20,
    ),
    TASK3=dict(
        name='VCR_QA-R',
        type='VL-logit',
        loss='CrossEntropyLoss',
        process='expand',
        loss_scale=1,
        task_id=7,
        dataroot=data_root + 'vcr/',
        features_h5path1=data_root + 'vcr/VCR_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'vcr/VCR_gt_resnet101_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath=data_root + 'vcr/train.jsonl',
        val_annotations_jsonpath=data_root + 'vcr/val.jsonl',
        max_seq_length=80,
        max_region_num=100,
        train_split='train',
        val_split='val',
        num_labels=1,
        lr=0.00002,
        per_gpu_train_batch_size=16,  # 16,
        per_gpu_eval_batch_size=16,  # 16,
        iters_in_epoch=13308,
        num_warmup_steps=13308,
        num_training_steps=266160,
        num_epoch=20,
    ),
    TASK4=dict(
        name='RetrievalFlickr30k',
        type='VL-logit',
        loss='CrossEntropyLoss',
        process='retrieval',
        loss_scale=1,
        task_id=9,
        dataroot=data_root + 'flickr30k/',
        features_h5path1=data_root + 'flickr30k/flickr30k_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2='',
        train_annotations_jsonpath=data_root + 'flickr30k/all_data_final_train_2014.jsonline',
        val_annotations_jsonpath=data_root + 'flickr30k/all_data_final_val_set0_2014.jsonline',
        max_seq_length=30,
        max_region_num=100,
        train_split='train',
        val_split='val',
        num_labels=1,
        lr=0.00002,
        per_gpu_train_batch_size=64,  # 64,
        per_gpu_eval_batch_size=64,  # 64,
        iters_in_epoch=470,  # todo
        num_warmup_steps=2196,
        num_training_steps=21960,
        num_epoch=20,
    ),
    TASK5=dict(
        name='refcoco+',
        type='V-logit',
        loss='BCEWithLogitLoss',
        process='normal',
        loss_scale=1,
        task_id=11,
        dataroot=data_root + 'referExpression/',
        features_h5path1=data_root + 'referExpression/refcoco+/refcoco+_resnet101_faster_rcnn_genome.lmdb',
        features_h5path2=data_root + 'referExpression/refcoco+/refcoco+_gt_resnet101_faster_rcnn_genome.lmdb',
        train_annotations_jsonpath='',
        val_annotations_jsonpath='',
        max_seq_length=20,
        max_region_num=100,
        train_split='train',
        val_split='val',
        num_labels=1,
        lr=0.00004,
        per_gpu_train_batch_size=32,  # 256,
        per_gpu_eval_batch_size=128,  # 1024,
        iters_in_epoch=470,
        num_warmup_steps=1410,
        num_training_steps=14100,
        num_epoch=30,
    ),
)
