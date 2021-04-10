# optimizer  transform.AdamW
optimizer = dict(
    type='Adam', lr=0.0002, weight_decay=0.0001,
    training_encoder_lr_multiply=1)  # mix_model_zrz_jin get_optimizer_parmeters??
optimizer_config = dict(grad_clip=None)  # ??
# fp16 = dict(
#     init_scale=2.**16,
#     growth_factor=2.0,
#     backoff_factor=0.5,
#     growth_interval=2000,
# )

# TODO(jinliang):copy zrz
# lr_config = dict(
#     use_warmup=True,
#     lr_steps=[90000, 108000],
#     lr_ratio=0.2,
#     warmup_factor=0.25,
#     warmup_iterations=10000,
#     policy='MultiStepScheduler')
lr_config = dict(
    policy='ReduceOnPlateauSchedule', use_warmup=False, factor=0.5, mode='max', patience=1, verbose=True, cooldown=2)
# max_iter = 118000
# max_iter = 236000
# by_iter = True
total_epochs = 30
