_base_ = [
    '../_base_/models/lxmert_pretrain_config.py',
    '../_base_/datasets/lxmert_pretrain_dataset.py',
    '../_base_/schedules/schedule_vqa.py',
    '../_base_/default_runtime.py'
]  # yapf:disable
total_epochs = 20
