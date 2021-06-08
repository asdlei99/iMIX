# To finetuning the base model with dense annotations
_base_ = [
    '../_base_/models/visual_dialog_bert_densen_anns_config.py',
    '../_base_/datasets/visual_dialog_dense_annotations_dataset.py',
    '../_base_/schedules/schedule_visual_dialog_dense.py',
    '../_base_/visual_dialog_bert_default_runtime.py'
]  # yapf:disable
