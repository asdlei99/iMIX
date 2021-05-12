custom_hooks = [
    dict(
        type='EMAIterHook',
        config=dict(
            use_ema=True,
            ema_decay_ratio=0.9999,
        ),
        level=30,  # NORMAL
    ),  # level type : PriorityStatus, str, int
    dict(
        type='EMAEpochHook',
        config=dict(
            use_ema=True,
            ema_decay_ratio=0.9999,
        ),
        level=30,
    ),
]
