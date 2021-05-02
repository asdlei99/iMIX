# model settings
weight_root = '/home/datasets/mix_data/iMIX/data/models/detectron.vmb_weights/'
model = dict(
    type='M4C',
    hidden_dim=768,
    dropout_prob=0.1,
    ocr_in_dim=3002,
    encoder=[
        dict(
            type='TextBertBase', text_bert_init_from_bert_base=True, hidden_size=768, params=dict(num_hidden_layers=3)),
        dict(
            type='ImageFeatureEncoder',
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file=weight_root + 'fc7_w.pkl',
            bias_file=weight_root + 'fc7_b.pkl',
        ),
        dict(
            type='ImageFeatureEncoder',
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file=weight_root + 'fc7_w.pkl',
            bias_file=weight_root + 'fc7_b.pkl',
        ),
    ],
    backbone=dict(type='MMT', hidden_size=768, num_hidden_layers=4),
    combine_model=dict(
        type='ModalCombineLayer',
        combine_type='non_linear_element_multiply',
        img_feat_dim=4096,
        txt_emb_dim=2048,
        dropout=0,
        hidden_dim=5000,
    ),
    head=dict(type='LinearHead', in_dim=768, out_dim=5000))

loss = dict(type='M4CDecodingBCEWithMaskLoss')
