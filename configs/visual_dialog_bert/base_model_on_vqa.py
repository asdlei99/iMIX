# To train the base model(no finetuning on dense annotations)
_base_ = [
    '../_base_/models/visual_dialog_bert_config.py',
    '../_base_/datasets/visual_dialog_dataset.py',
    '../_base_/schedules/schedule_visual_dialog.py',
    '../_base_/default_runtime.py'
]  # yapf:disable
