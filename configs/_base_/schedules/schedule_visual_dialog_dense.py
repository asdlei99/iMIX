# optimizer  transform.AdamW
# optimizer = dict(
#     type='AdamW',
#     lr=0.00005,
#     weight_decay=0,
#     eps=1e-9,
#     betas=[0.9, 0.98],
#     training_encoder_lr_multiply=1
# )  # mix_model_zrz_jin get_optimizer_parmeters??

optimizer = dict(
    type='AdamW',
    constructor='BertOptimizerConstructor',
    paramwise_cfg=dict(
        language_weights_file='~/iMIX/imix/imix/models/visual_dialog_model/config/language_weights.json'),
    lr=1e-4,  # 2e-5
    image_lr=1e-4,  # learning rate for vision params
    # weight_decay=0,
    # eps=1e-9,
    # betas=[0.9, 0.98],
    training_encoder_lr_multiply=1)
optimizer_config = dict(grad_clip=None)  # ??
# fp16 = dict(
#     init_scale=2.**16,
#     growth_factor=2.0,
#     backoff_factor=0.5,
#     growth_interval=2000,
# )

lr_config = dict(
    policy='WarmupLinearScheduleNonZero',
    use_warmup=True,
    warmup_iterations=10000,  # 10000
    t_total=200000,  # 200000
)
# max_iter = 118000
# max_iter = 236000
# by_iter = True
total_epochs = 20
