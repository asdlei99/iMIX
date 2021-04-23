# model settings
model = dict(
    type='LXMERT',
    params=dict(
        random_initialize=False,
        num_labels=3129,
        # BertConfig
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        #
        mode='lxr',
        l_layers=9,  # 12
        x_layers=5,  # 5
        r_layers=5,  # 0
        visual_feat_dim=2048,
        visual_pos_dim=4,
        freeze_base=False,
        max_seq_length=20,
        seed=9595,
        model='bert',
        training_head_type='vqa2',
        bert_model_name='bert-base-uncased',
        pretrained_path='/home/datasets/mix_data/iMIX/data/models/model_LXRT.pth',
        label2ans_path='/home/datasets/mix_data/lxmert/vqa/trainval_label2ans.json',
        # for pretraining
        # gqa_label=1534,
        # task_matched=False,
        # task_mask_lm=False,
        # task_obj_predict=False,
        # task_qa=False,
        # word_mask_rate=0.15,
        # obj_mask_rate=0.15,
        # qa_sets=None,
        # visual_losses='obj,attr,feat',
    ))

loss = dict(type='LogitBinaryCrossEntropy')
