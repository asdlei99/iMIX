# # 前傳沒問題
# _base_ = [
#     '../_base_/models/lxmert_config.py',
#     '../_base_/datasets/vqa_dataset.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ]  # yapf:disable

# To train the base model(no finetuning on dense annotations)
# _base_ = [
#     '../_base_/models/visual_dialog_bert_config.py',
#     '../_base_/datasets/visual_dialog_dataset.py',
#     '../_base_/schedules/schedule_visual_dialog.py',
#     '../_base_/default_runtime.py'
# ]  # yapf:disable

# To finetuning the base model with dense annotations
# _base_ = [
#     '../_base_/models/visual_dialog_bert_densen_anns_config.py',
#     '../_base_/datasets/visual_dialog_dense_annotations_dataset.py',
#     '../_base_/schedules/schedule_visual_dialog_dense.py',
#     '../_base_/visual_dialog_bert_default_runtime.py'
# ]  # yapf:disable

# To finetuning the base model with dense annotations and the next sentence prediction(NSP) loss
# _base_ = [
#     '../_base_/models/visual_dialog_bert_densen_anns_config_ce+nsp.py',
#     '../_base_/datasets/visual_dialog_dense_annotations_dataset.py',
#     '../_base_/schedules/schedule_visual_dialog_dense.py',
#     '../_base_/visual_dialog_bert_default_runtime.py'
# ]  # yapf:disable

# 前傳沒問題  TODO(zhaojian)
# _base_ = [
#     '../_base_/models/vilbert_config.py',
#     '../_base_/datasets/vqa_dataset.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ] # yapf:disable

# 前傳沒問題
# _base_ = [
#     '../_base_/models/visualbert_config.py',
#     '../_base_/datasets/vqa_dataset.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ] # yapf:disable

# 前傳沒問題
# _base_ = [
#     '../_base_/models/cmrin_config.py',
#     '../_base_/datasets/refcoco_dataset.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ] # yapf:disable

# 前傳沒問題
_base_ = [
    '../_base_/models/m4c_config.py',
    '../_base_/datasets/textvqa_dataset.py',
    '../_base_/schedules/schedule_vqa.py',
    '../_base_/default_runtime.py'
]  # yapf:disable

# 前傳沒問題
# _base_ = [
#     '../_base_/models/r2c_config.py',
#     '../_base_/datasets/vcr_dataset.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ]  # yapf:disable

# 前傳沒問題  chao
# _base_ = [
#     '../_base_/models/hgl_config.py',
#     '../_base_/datasets/vcr_dataset.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ] # yapf:disable

# 前傳沒問題
# _base_ = [
#     '../_base_/models/pythia_config.py',
#     '../_base_/datasets/vqa_dataset_w_global.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ] # yapf:disable

# 前傳沒問題  chao
# _base_ = [
#     '../_base_/models/lcgn_config.py',
#     '../_base_/datasets/gqa_dataset.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ]  # yapf:disable

# 前傳沒問題
# _base_ = [
#     '../_base_/models/ban_config.py',
#     '../_base_/datasets/vqa_dataset.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ] # yapf:disable

# 不通过
# _base_ = [
#     '../_base_/models/mcan_config.py',
#     '../_base_/datasets/vqa_dataset_grid_data.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ]  # yapf:disable

# 前傳沒問題
# _base_ = [
#     '../_base_/models/resc_config.py',
#     '../_base_/datasets/refcoco_dataset.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ] # yapf:disable

# 前傳沒問題 UNITER
# _base_ = [
#     '../_base_/models/uniter_config.py',
#     '../_base_/datasets/vqa_dataset_uniter.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ] # yapf:disable

# 前傳沒問題 pretrain for lxmert
# _base_ = [
#     '../_base_/models/lxmert_pretrain_config.py',
#     '../_base_/datasets/lxmert_pretrain_dataset.py',
#     '../_base_/schedules/schedule_vqa.py',
#     '../_base_/default_runtime.py'
# ] # yapf:disable

# _base_ = [
#     '../_base_/models/devlbert_config.py',
#     '../_base_/datasets/devlbert_dataset.py',
#     '../_base_/schedules/schedule_vqa_devlbert.py',
#     '../_base_/default_runtime.py'
# ]
# total_epochs = 20
