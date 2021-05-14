import torch
from imix.models.builder import build_embedding, build_encoder
from imix.models.vqa_models.base_model import UniterBaseModel


# @VQA_MODELS.register_module()
class UNITER(UniterBaseModel):

    def __init__(self, embeddings, encoder, **kwargs):
        super().__init__()

        self.embedding = build_embedding(embeddings)
        self.encoder = build_encoder(encoder)
        # self.head = build_head(head)

        # pretrained_path = kwargs['pretrained_path']
        # ckpt = torch.load(pretrained_path)
        # self.load_my_state_dict(ckpt)

    # not load head weights from pretrained weights
    def load_my_state_dict(self, pretrained_path):
        state_dict = torch.load(pretrained_path)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        old_keys = []
        new_keys = []
        for name in state_dict.keys():
            new_key = None
            if 'uniter.embeddings' in name:
                new_key = name.replace('uniter.embeddings', 'embedding.0')
            if 'uniter.img_embeddings' in name:
                new_key = name.replace('uniter.img_embeddings', 'embedding.1')
            if 'uniter.encoder' in name:
                new_key = name.replace('uniter.encoder', 'encoder')
            if 'uniter.pooler' in name:
                new_key = name.replace('uniter.pooler', 'encoder.pooler')
            if new_key:
                old_keys.append(name)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(self, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(self, prefix=start_prefix)
        # for name, param in state_dict.items():
        #     if 'uniter.embeddings' in name:
        #         own_state[name.replace('uniter.embeddings', 'embedding.0')].copy_(param)
        #     elif 'uniter.img_embeddings' in name:
        #         own_state[name.replace('uniter.img_embeddings', 'embedding.1')].copy_(param)
        #     elif 'uniter.encoder' in name:
        #         own_state[name.replace('uniter.encoder', 'encoder')].copy_(param)
        #     elif 'uniter.pooler' in name:
        #         own_state[name.replace('uniter.pooler', 'encoder.pooler')].copy_(param)
        #     if name not in own_state:
        #         continue
        #     own_state[name].copy_(param)

    def _compute_txt_embeddings(self, input_ids, position_ids, txt_type_ids=None):
        output = self.embedding[0](input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None, img_type_ids=None):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embedding[0].token_type_embeddings(img_type_ids)
        output = self.embedding[1](img_feat, img_pos_feat, img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self,
                                    input_ids,
                                    position_ids,
                                    img_feat,
                                    img_pos_feat,
                                    gather_index,
                                    img_masks=None,
                                    txt_type_ids=None,
                                    img_type_ids=None):
        txt_emb = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(img_feat, img_pos_feat, img_masks, img_type_ids)
        if gather_index is not None:
            # align back to most compact input
            gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.encoder.config.hidden_size)
            embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1), dim=1, index=gather_index)
        else:
            embedding_output = torch.cat([txt_emb, img_emb], dim=1)

        return embedding_output

    def forward_train(self, data, output_all_encoded_layers=False, **kwargs):
        input_ids = data['input_ids']
        position_ids = data['position_ids']
        img_feat = data['img_feat']
        img_pos_feat = data['img_pos_feat']
        attn_masks = data['attn_masks']
        gather_index = data['gather_index']
        img_masks = data['img_masks']
        txt_type_ids = data['txt_type_ids']
        img_type_ids = data['img_type_ids']

        extended_attention_mask = attn_masks.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        else:
            embedding_output = self._compute_img_txt_embeddings(input_ids, position_ids, img_feat, img_pos_feat,
                                                                gather_index, img_masks, txt_type_ids, img_type_ids)

        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers
        # pooled_output = self.encoder.pooler(encoded_layers)
        # return pooled_output
        # logits = self.head(pooled_output)
        # try:
        #     model_outputs = {'scores': logits, 'target': data['answers_scores'].cuda()}
        # except Exception:
        #     pass
        # return model_outputs

    def forward_test(self, data, **kwargs):
        model_output = self.forward_train(data)
        return model_output
