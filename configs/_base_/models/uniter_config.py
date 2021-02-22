# model settings
model = dict(
    type='UNITER',
    embeddings=[
        dict(
            type='UniterTextEmbeddings',
            vocab_size=28996,
            hidden_size=768,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.1),
        dict(
            type='UniterImageEmbeddings',
            img_dim=2048,
            hidden_size=768,
            hidden_dropout_prob=0.1)
    ],
    encoder=dict(
        type='UniterEncoder',
        config_file='configs/_base_/models/uniter-base.json'),
    head=dict(
        type='UNITERHead',
        in_dim=768,
        out_dim=3129
    ),
    pretrained_path='/home/datasets/UNITER/uniter-base.pt'
)
loss=dict(type='LogitBinaryCrossEntropy')
