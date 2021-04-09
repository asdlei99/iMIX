# model settings
model = dict(
    type='LXMERT',
    params=dict(
        bert_model_name='bert-base-uncased',
        training_head_type='gqa',
        random_initialize=False,
        num_labels=1842,
        gqa_labels=1842,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_size=768,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        layer_norm_eps=1e-12,
        mode='lxr',
        l_layers=9,  # 12
        x_layers=5,  # 5
        r_layers=5,  # 0
        visual_feat_dim=2048,
        visual_pos_dim=4,
        task_matched=True,
        task_mask_lm=True,
        task_obj_predict=True,
        task_qa=True,
        special_visual_initialize=True,  # i dont know what this is
        hard_cap_seq_len=36,
        cut_first='text',
        embedding_strategy='plain',
        bypass_transformer=False,
        output_attentions=False,  # need to implement
        output_hidden_states=False,  # need to implement
        text_only=False,
        freeze_base=False,
        finetune_lr_multiplier=1,
        vocab_size=30522,
        fast_mode=False,
        dynamic_attention=False,  # need to implement
        in_batch_pairs=False,
        visualization=False,  # need to implement
        max_seq_length=20,
        model='bert',
        pretrained_path='/home/datasets/mix_data/mmf/data/models/model_LXRT.pth'
    ))


loss = dict(type='LogitBinaryCrossEntropy')
