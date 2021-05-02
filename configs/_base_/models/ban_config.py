# model settings
model = dict(
    type='BAN',
    embedding=[
        dict(
            type='WordEmbedding',
            vocab_file='/home/datasets/mix_data/iMIX/data/datasets/textvqa/defaults/extras/vocabs/vocabulary_100k.txt',
            embedding_dim=300,
            glove_params=dict(
                name='6B',
                dim=300,
                cache='/home/datasets/mix_data/iMIX',
            )),
        dict(
            type='BiLSTMTextEmbedding',
            hidden_dim=1280,
            num_layers=1,
            dropout=0,
            embedding_dim=300,
        )
    ],
    backbone=dict(
        type='BAN_BACKBONE',
        v_dim=2048,
        num_hidden=1280,
        gamma=4,
        k=1,
        activation='ReLU',
        dropout=0.2,
    ),
    head=dict(
        type='WeightNormClassifierHead',
        in_dim=1280,
        out_dim=3129,
        hidden_dim=2560,
        dropout=0.5,
        # loss_cls=dict(type='LogitBinaryCrossEntropy')
    ))

loss = dict(type='LogitBinaryCrossEntropy')
