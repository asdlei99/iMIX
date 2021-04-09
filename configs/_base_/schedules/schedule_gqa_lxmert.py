# optimizer modified transform.AdamW
optimizer = dict(
    type='BertAdam',
    lr=1e-5,
    weight_decay=0.01,
    eps=1e-6,
    betas=[0.9, 0.999],
    training_encoder_lr_multiply=1
)
optimizer_config = dict(grad_clip=dict(max_norm=5))
#optimizer_config = dict(grad_clip=None)
'''
fp16 = dict(
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)
'''
lr_config = dict(
    warmup=0.1,
    warmup_method='warmup_linear',
    #max_iters=117876,  # ceil(totoal 942999 / batch size 32) * epoch size datasets: train
    max_iters=134380,  # floor(totoal 1075062 / batch size 32) * epoch size datasets: train, valid
    policy='BertWarmupLinearLR')

#by_iter = True
total_epochs = 4
