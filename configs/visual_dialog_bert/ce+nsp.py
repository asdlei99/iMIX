# To finetuning the base model with dense annotations and the next sentence prediction(NSP) loss
_base_ = [
    '../_base_/models/visual_dialog_bert_densen_anns_config_ce+nsp.py',
    '../_base_/datasets/visual_dialog_dense_annotations_dataset.py',
    '../_base_/schedules/schedule_visual_dialog_dense.py',
    '../_base_/visual_dialog_bert_default_runtime.py'
]  # yapf:disable
