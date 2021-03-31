import torch
import torch.nn as nn
from apex.normalization.fused_layer_norm import FusedLayerNorm

from ..builder import EMBEDDING


@EMBEDDING.register_module()
class UniterTextEmbeddings(nn.Module):

    def __init__(self,
                 vocab_size=28996,
                 hidden_size=768,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 hidden_dropout_prob=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings + position_embeddings + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@EMBEDDING.register_module()
class UniterImageEmbeddings(nn.Module):

    def __init__(self, img_dim=2048, hidden_size=768, hidden_dropout_prob=0.1):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, hidden_size)
        self.img_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.pos_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(7, hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
