# config_file = {
#     "attention_probs_dropout_prob": 0.1,
#     "bi_attention_type": 1,
#     "bi_hidden_size": 1024,
#     "bi_intermediate_size": 1024,
#     "bi_num_attention_heads": 8,
#     "fast_mode": False,
#     "fixed_t_layer": 0,
#     "fixed_v_layer": 0,
#     "fusion_method": "mul",
#     "hidden_act": "gelu",
#     "hidden_dropout_prob": 0.1,
#     "hidden_size": 768,
#     "in_batch_pairs": False,
#     "initializer_range": 0.02,
#     "intermediate_size": 3072,
#     "intra_gate": False,
#     "max_position_embeddings": 512,
#     "num_attention_heads": 12,
#     "num_hidden_layers": 12,
#     "pooling_method": "mul",
#     "predict_feature": False,
#     "t_biattention_id": [
#         6,
#         7,
#         8,
#         9,
#         10,
#         11
#     ],
#     "type_vocab_size": 2,
#     "v_attention_probs_dropout_prob": 0.1,
#     "v_biattention_id": [
#         0,
#         1,
#         2,
#         3,
#         4,
#         5
#     ],
#     "v_feature_size": 2048,
#     "v_hidden_act": "gelu",
#     "v_hidden_dropout_prob": 0.1,
#     "v_hidden_size": 1024,
#     "v_initializer_range": 0.02,
#     "v_intermediate_size": 1024,
#     "v_num_attention_heads": 8,
#     "v_num_hidden_layers": 6,
#     "v_target_size": 1601,
#     "vocab_size": 30522,
#     "with_coattention": True
# }
#
# # model settings
# model = dict(
#     type='VisDiaBERT',
#     embeddings=[
#         dict(
#             type='VisDiaBertEmbeddingsDialog',
#             config=config_file
#         ),
#         dict(
#             type='VisDiaBertImageEmbeddings',
#             config=config_file,
#         )
#     ],
#
#     encoder=dict(
#         type='VisDiaBertEncoder',
#         config=config_file,
#     ),
#     pooler=[
#         dict(
#             type='VisDiaBertTextPooler',
#             config=config_file,
#         ),
#         dict(
#             type='VisDiaBertImagePooler',
#             config=config_file,
#         ),
#     ],
#     head=dict(
#         type='VisDiaBertPreTrainingHeads',
#         config=config_file,
#         # text_prediction=dict(type='BertLMPredictionHead', config=config_file, ),
#         # image_Predictions=dict(type='BertImagePredictionHead', config=config_file, ),
#         # bi_seq_relationship=dict(bi_hidden_size=1, output_size=2),
#         # fusion_method='sum',
#         # dropout_probability=0.1,
#     ),
#     pretrained_path='/home/datasets/UNITER/uniter-base.pt',
# )
# loss = [
#     dict(type=''),
#     dict(type=''),
#     dict(type=''),
# ]

model = dict(
    type='VisDiaBERT',
    config=dict(
        pretrained_model_name_or_path='/home/datasets/mix_data/torch/pytorch_transformers/bert/bert-base-uncased',
        bert_file_path='~/iMIX/imix/imix/models/visual_dialog_model/config/bert_base_6layer_6conect.json',
        sample_size=10,  # equal to batch_size*2
        is_dense=True,  # dense annotations -> sample_size = 80
    ))

# loss = [
#     dict(type='KLDivLoss'),
#     dict(type='CrossEntropyLoss'),
#     dict(type='CrossEntropyLoss'),
# ]
# loss = dict(
#     type='VisualDialogBertLoss',
#     MLM_loss=dict(type='CrossEntropyLoss', weight_coeff=1, params=dict(ignore_index=-1)),
#     # masked language modeling loss
#     NSP_loss=dict(type='CrossEntropyLoss', weight_coeff=1, params=dict(ignore_index=-1)),
#     # next sentence prediction loss
#     MIR_loss=dict(type='KLDivLoss', weight_coeff=1, params=dict(reduction='none')),  # masked image region loss
# )

loss = dict(
    type='VisualDialogBertDenseLoss',
    NSP_loss=dict(type='CrossEntropyLoss', weight_coeff=1),
    KLDiv_loss=dict(type='KLDivLoss', weight_coeff=1, params=dict(reduction='batchmean')),
    MLM_loss=dict(type='CrossEntropyLoss', weight_coeff=0.01, params=dict(ignore_index=-1)),
    MIR_loss=dict(type='KLDivLoss', weight_coeff=0.01, params=dict(reduction='none')),  # masked image region loss
)

# loss = dict(type='BinaryCrossEntropyWithLogits')
