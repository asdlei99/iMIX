# model settings
model = dict(
    type='PYTHIA',
    embedding=[
        dict(
            type='WordEmbedding',
            vocab_file='/home/zrz/.cache/torch/iMIX/data/datasets/textvqa/defaults/extras/vocabs/vocabulary_100k.txt',
            embedding_dim=300),
        dict(
            type='TextEmbedding',
            emb_type='attention',
            hidden_dim=1024,
            num_layers=1,
            conv1_out=512,
            conv2_out=2,
            dropout=0,
            embedding_dim=300,
            kernel_size=1,
            padding=0)
    ],
    encoder=[
        dict(
            type='ImageFeatureEncoder',
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='/home/zrz/.cache/torch/iMIX/data/models/detectron.vmb_weights/fc7_w.pkl',
            bias_file='/home/zrz/.cache/torch/iMIX/data/models/detectron.vmb_weights/fc7_b.pkl',
        ),
        dict(type='ImageFeatureEncoder', encoder_type='default'),
    ],
    backbone=[
        dict(
            type='ImageFeatureEmbedding',
            img_dim=2048,
            question_dim=2048,
            modal_combine=dict(type='non_linear_element_multiply', params=dict(dropout=0, hidden_dim=5000)),
            normalization='softmax',
            transform=dict(type='linear', params=dict(out_dim=1))),
        dict(
            type='ImageFeatureEmbedding',
            img_dim=2048,
            question_dim=2048,
            modal_combine=dict(type='non_linear_element_multiply', params=dict(dropout=0, hidden_dim=5000)),
            normalization='softmax',
            transform=dict(type='linear', params=dict(out_dim=1))),
    ],
    combine_model=dict(
        type='ModalCombineLayer',
        combine_type='non_linear_element_multiply',
        img_feat_dim=4096,
        txt_emb_dim=2048,
        dropout=0,
        hidden_dim=5000,
    ),
    head=dict(type='LogitClassifierHead', in_dim=5000, out_dim=3129, img_hidden_dim=5000, text_hidden_dim=300))
loss = dict(type='LogitBinaryCrossEntropy')
