# optimizer  transform.AdamW
optimizer = dict(
    type='UniterAdamW', lr=6e-5, weight_decay=0.01, eps=1e-6, betas=[0.9, 0.98], training_encoder_lr_multiply=1)
optimizer_config = dict(grad_clip=None)  # ??
fp16 = dict(
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)

# learning policy
lr_config = dict(
    use_warmup=True,
    lr_steps=[90000, 108000],
    lr_ratio=3e-05,
    warmup_factor=0.25,
    warmup_iterations=800,
    policy='MultiStepScheduler')

total_epochs = 8
