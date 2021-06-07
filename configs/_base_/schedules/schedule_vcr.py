# optimizer  transform.AdamW
optimizer = dict(type='Adam', lr=0.0002, weight_decay=0.0001, training_encoder_lr_multiply=1)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='ReduceOnPlateauSchedule', use_warmup=False, factor=0.5, mode='max', patience=1, verbose=True, cooldown=2)

total_epochs = 30
