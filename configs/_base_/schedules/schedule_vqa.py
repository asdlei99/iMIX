# optimizer  transform.AdamW
optimizer = dict(
    type='AdamW', lr=0.00005, weight_decay=0, eps=1e-9,
    betas=[0.9, 0.98])  # mix_model_zrz_jin get_optimizer_parmeters??
optimizer_config = dict(grad_clip=None)  # ??
# learning policy
lr_config = dict(
    policy='step',  # TODO(jinliang) multiStep  <-  step
    warmup='linear',  # TODO(jinliang) mix_model_zrz_jin warmup -> linear??
    warmup_iters=27000,
    warmup_ratio=0.25,
    step=[90000, 108000])
max_iter = 118000
