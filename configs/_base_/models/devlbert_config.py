# model settings
DATA_ROOT = '/home/datasets/mix_data/DeVLBert'
model = dict(
    type='DeVLBert',
    config=dict(
        pretrained_model_name_or_path=DATA_ROOT + '/vqa/coco/pre_trained/vqa-pytorch_model_13_ema.bin',
        bert_file_path=DATA_ROOT + '/bert_file/bert_base_6layer_6conect.json',
        num_labels=3129,
        default_gpu=True))
loss = dict(type='BCEWithLogitsLoss')
