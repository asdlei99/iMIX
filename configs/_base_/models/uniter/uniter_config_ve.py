# model settings for VE
model = dict(
    type='UNITER_VE',
    params=dict(
        model_config='configs/_base_/models/uniter/uniter-base.json',
        gradient_accumulation_steps=4,
        dropout=0.1,
        seed=42,
        img_dim=2048,
        pretrained_path='/home/datasets/mix_data/UNITER/uniter-base.pt',
    ),
)

loss = dict(type='LogitBinaryCrossEntropy')

optimizer = dict(
    type='TansformerAdamW',
    constructor='UniterOptimizerConstructor',
    paramwise_cfg=dict(weight_decay=0.01),
    lr=8e-5,
    betas=[0.9, 0.98],
    training_encoder_lr_multiply=1,
)
optimizer_config = dict(grad_clip=dict(max_norm=2.0))

# fp16 = dict(
#     init_scale=2.**16,
#     growth_factor=2.0,
#     backoff_factor=0.5,
#     growth_interval=2000,
# )

lr_config = dict(
    num_warmup_steps=400,
    num_training_steps=4000,
    policy='WarmupLinearSchedule',
)

total_epochs = 6

eval_iter_period = 500
checkpoint_config = dict(iter_period=eval_iter_period)

gradient_accumulation_steps = 4
is_lr_accumulation = True
