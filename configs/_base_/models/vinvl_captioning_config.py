# model settings
model = dict(
    type='OSCAR',
    params=dict(
        config_name=None,
        drop_out=0.1,
        img_feature_dim=2054,
        img_feature_type='frcnn',
        length_penalty=1,
        loss_type='sfmx',
        min_constraints_to_satisfy=2,
        model_name_or_path='/home/datasets/mix_data/model/oscar/base-vg-labels/ep_67_588997/',
        num_beams=5,
        num_keep_best=1,
        num_labels=2,
        num_return_sequences=1,
        output_hidden_states=False,
        output_mode='classification',
        repetition_penalty=1,
        scst=False,
        seed=88,
        temperature=1,
        tokenizer_name=None,
        top_k=0,
        top_p=1,
        use_cbs=False,
        tie_weights=True,
        freeze_embedding=True,
        label_smoothing=0.1,
        drop_worst_ratio=0.2,
        drop_worst_after=20000,
        training_head_type='vqa2',
        bert_model_name='bert-base-uncased',
        # no_cuda=False,
        # gradient_accumulation_steps=1,
    ))

loss = dict(
    type='OSCARLoss', cfg=dict(
        loss_type='sfmx',
        ngpu=1,
        num_labels=3129,
        gradient_accumulation_steps=1,
    ))

optimizer = dict(
    type='TansformerAdamW',
    constructor='OscarOptimizerConstructor',
    paramwise_cfg=dict(weight_decay=0.05, ),
    lr=3e-05,
    eps=1e-8,
    training_encoder_lr_multiply=1,
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

lr_config = dict(
    num_warmup_steps=0,  # warmup_proportion=0
    num_training_steps=123950,  # ceil(totoal 634516 / batch size 32 / GPUS 4) * epoch size 25
    policy='WarmupLinearSchedule',
)

# by_iter = True
total_epochs = 30
