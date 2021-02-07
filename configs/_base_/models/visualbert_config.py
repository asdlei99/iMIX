# model settings
model = dict(
    type='VisualBERT',
    params=dict(
        output_attentions=False,
        output_hidden_states=False,
        pooler_strategy='default',
        bert_model_name='bert-base-uncased',
        visual_embedding_dim=2048,
        embedding_strategy='plain',
        bypass_transformer=False,
        num_labels=3129,
        training_head_type='hateful_memes',
        special_visual_initialize=True,
        freeze_base=False,
        random_initialize=False))

loss = dict(type='BinaryCrossEntropyWithLogits')
