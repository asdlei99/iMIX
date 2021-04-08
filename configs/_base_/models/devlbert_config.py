# model settings
model = dict(
    type='DeVLBert',
    config=dict(
        pretrained_model_name_or_path = '/home/wbq/Desktop/0323/DeVLBert-master/vqa-pytorch_model_13_ema.bin',
        bert_file_path = '/home/wbq/Desktop/0323/DeVLBert-master/config/bert_base_6layer_6conect.json',
        num_labels=3129,
        default_gpu=True
    ))

loss = dict(type='BCEWithLogitsLoss')
