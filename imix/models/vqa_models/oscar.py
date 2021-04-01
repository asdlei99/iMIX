import torch

from ..builder import VQA_MODELS, build_embedding, build_encoder
from .base_model import BaseModel


@VQA_MODELS.register_module()
class OSCAR(BaseModel):

    def __init__(self, embedding, encoder):
        super().__init__()

        # params = kwargs['params']
        # self.special_visual_initialize = params['special_visual_initialize']
        # freeze_base = params['freeze_base']
        # self.training_head_type = params['training_head_type']
        # if self.training_head_type == 'pretraining':
        #     self.model = VisualBERTForPretraining(**params)
        # else:
        #     self.model = VisualBERTForClassification(**params)

        self.embedding = build_embedding(embedding)
        self.model = build_encoder(encoder)

        if self.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        if freeze_base:
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def forward(self, data):

        input_ids = data.input_ids
        position_ids = data.position_ids
        token_type_ids = data.token_type_ids
        img_feats = data.img_feats

        embedding_output = self.embedding_model.text_embedding(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        if encoder_history_states:
            assert img_feats is None, 'Cannot take image features while using encoder history states'

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.embedding_model.code_embeddings(img_feats)
                img_embedding_output = self.embedding_model.image_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t':  # transpose
                code_emb = self.embedding_model.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.embedding_model.image_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale':  # left scaled
                code_emb = self.embedding_model.code_embeddings(img_feats)
                img_embedding_output = self.embedding_model.image_embedding(code_emb)
            else:
                img_embedding_output = self.embedding_model.image_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)

        encoder_outputs, pooled_output = self.model(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            encoder_history_states=encoder_history_states)
        # sequence_output = encoder_outputs[0]

        # add hidden_states and attentions if they are here
        # outputs = (
        #               encoder_outputs[0],
        #               pooled_output,
        #           ) + encoder_outputs[1:]

        logits = self.head(pooled_output)
        return logits
