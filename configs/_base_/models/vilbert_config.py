# model settings
tasks = '11'  # '1-2-3...' training task separate by -
'''
num_labels=3129,    # for TASK1
num_labels=3129,    # for TASK2
num_labels=,        # for TASK3
num_labels=1,       # for TASK4
num_labels=,        # for TASK5
num_labels=,        # for TASK6
num_labels=1,       # for TASK7
num_labels=1,       # for TASK8
num_labels=1,       # for TASK9
num_labels=1,       # for TASK10
num_labels=1,       # for TASK11
num_labels=,        # for TASK12
num_labels=3,       # for TASK13
num_labels=3,       # for TASK14
num_labels=1533,    # for TASK15
num_labels=,        # for TASK16
num_labels=1,       # for TASK17
num_labels=,        # for TASK18
'''

model = dict(
    type='VILBERT',
    params=dict(
        num_labels=1,
        # below from bert_base_6layer_6conect.json
        bi_hidden_size=1024,
        bi_num_attention_heads=8,
        bi_intermediate_size=1024,
        bi_attention_type=1,
        pooling_method='mul',
        visual_target=0,  # which target to use for visual branch 0: soft label, 1: regress the feature, 2: NCE loss."
        fast_mode=False,
        fixed_v_layer=0,
        fixed_t_layer=0,
        in_batch_pairs=False,
        fusion_method='mul',
        dynamic_attention=False,
        with_coattention=True,
        objective=0,
        num_negative=128,
        model='bert',
        task_specific_tokens=True,
        visualization=False,
        t_config=dict(
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            max_position_embeddings=512,
            num_attention_heads=12,
            num_hidden_layers=12,
            type_vocab_size=2,
            vocab_size=30522,
            t_biattention_id=[6, 7, 8, 9, 10, 11],
            layer_norm_eps=1e-12,
            task_specific_tokens=True,
        ),
        v_config=dict(
            feature_size=2048,
            target_size=1601,
            hidden_size=1024,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            initializer_range=0.02,
            biattention_id=[0, 1, 2, 3, 4, 5],
        ),
        # below from parse argument
        tasks=tasks,  # '1-2-3...' training task separate by -
        bert_model='bert-base-uncased',
        from_pretrained='/home/datasets/mix_data/model/vilbert/multi_task_model.bin',
        train_iter_multiplier=1,  # multiplier for the multi-task training
        # forward every n iteration is the validation score is not improving over the last 3 epoch, -1 means will stop
        train_iter_gap=4,
        do_lower_case=True,
        seed=0,
        gradient_accumulation_steps=1,
        freeze=-1,  # till which layer of textual stream of vilbert need to fixed
        vision_scratch=False,  # whether pre-trained the image or not.
        baseline=False,  # whether use single stream baseline
        fp16=False,  # Whether to use 16-bit float precision instead of 32-bit
        # resume_file= ,# Resume from checkpoint
        # below for imix
        training_head_type='vqa2',
    ))

loss = dict(
    type='VILBERTMutilLoss',
    task_cfg=dict(
        tasks=tasks,
        gradient_accumulation_steps=1,
        TASK1=dict(
            loss='BCEWithLogitLoss',
            loss_scale=1,
            type='VL-classifier',
        ),
        TASK2=dict(
            type='VL-classifier',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
        TASK3=dict(
            type='VL-logit',
            loss='CrossEntropyLoss',
            loss_scale=1,
        ),
        TASK4=dict(
            type='V-logit-mc',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
        TASK5=dict(
            type='VL-logit',
            loss='CrossEntropyLoss',
            loss_scale=1,
        ),
        TASK6=dict(
            type='VL-logit',
            loss='CrossEntropyLoss',
            loss_scale=1,
        ),
        TASK7=dict(
            type='VL-logit',
            loss='CrossEntropyLoss',
            loss_scale=1,
        ),
        TASK8=dict(
            type='VL-logit',
            loss='CrossEntropyLoss',
            loss_scale=1,
        ),
        TASK9=dict(
            type='V-logit',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
        TASK10=dict(
            type='V-logit',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
        TASK11=dict(
            type='V-logit',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
        TASK12=dict(
            type='VL-binary-classifier',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
        TASK13=dict(
            type='VL-tri-classifier',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
        TASK14=dict(
            type='VL-tri-classifier',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
        TASK15=dict(
            type='VL-classifier-GQA',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
        TASK16=dict(
            type='VL-binary-classifier',
            loss='CrossEntropyLoss',
            loss_scale=1,
        ),
        TASK17=dict(
            type='V-logit-mc',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
        TASK18=dict(
            type='V-logit',
            loss='BCEWithLogitLoss',
            loss_scale=1,
        ),
    ))

optimizer = dict(
    type='TansformerAdamW',
    constructor='VilbertOptimizerConstructor',
    paramwise_cfg=dict(
        language_weights_file='/home/datasets/mix_data/model/vilbert/config/bert-base-uncased_weight_name.json',
        vision_scratch=False,  # whether pre-trained the image or not.
    ),
    lr=0.00002,
    correct_bias=False,
    training_encoder_lr_multiply=1)
optimizer_config = dict(grad_clip=None)
'''
lr=0.00004,# for TASK1
lr=0.00004,# for TASK2
lr=0.00004,# for TASK3
lr=0.00002,# for TASK4
lr=0.00002,# for TASK5
lr=0.00002,# for TASK6
lr=0.00002,# for TASK7
lr=0.00002,# for TASK8
lr=0.00002,# for TASK9
lr=0.00002,# for TASK10
lr=0.00002,# for TASK11
lr=0.00002,# for TASK12
lr=0.00002,# for TASK13
lr=0.00004,# for TASK14
lr=0.00004,# for TASK15
lr=0.00004,# for TASK16
lr=0.00002,# for TASK17
lr=0.000002,# for TASK18
'''

lr_config = dict(
    num_warmup_steps=512,  # warmup_proportion=0.1
    num_training_steps=5120,  # ceil(totoal 443753 / batch size 32) * epoch size
    policy='WarmupLinearSchedule')
'''
# for task 1
num_warmup_steps=8472,
num_training_steps=84720,
# for task 2
num_warmup_steps=20224,
num_training_steps=202240,
# for task 4
num_warmup_steps=734,
num_training_steps=7340,
# for task 7
num_warmup_steps=7620,
num_training_steps=76200,
# for task 8
num_warmup_steps=2196,
num_training_steps=21960,
# for task 9
num_warmup_steps=752,
num_training_steps=7520,
# for task 10
num_warmup_steps=750,
num_training_steps=7500,
# for task 11
num_warmup_steps=512,
num_training_steps=5120,
# for task 13
num_warmup_steps=4004,
num_training_steps=40040,
# for task 14
num_warmup_steps=4530,
num_training_steps=45300,
# for task 15
num_warmup_steps=15046,
num_training_steps=150460,
# for task 17
num_warmup_steps=3138,
num_training_steps=31380,
'''

# by_iter = True
total_epochs = 20
'''
fp16 = dict(
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)

lr_config = dict(
    num_warmup_steps= ,
    policy='WarmupConstantSchedule')

lr_config = dict(
    mode="max",
    factor=0.2,
    patience=1,
    cooldown=1,
    threshold=0.001,
    policy='ReduceLROnPlateau')

lr_config = dict(
    T_max=, #median_num_iter * config[total_epochs]
    policy='CosineAnnealingLR')


lr_config = dict(
    T_0=, #median_num_iter * config[total_epochs]
    policy='CosineAnnealingWarmRestarts')
'''
