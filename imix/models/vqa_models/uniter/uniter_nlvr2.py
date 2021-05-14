from imix.models.builder import VQA_MODELS
from imix.models.vqa_models.base_model import UniterBaseModel
from .uniter import UNITER
from .attention import MultiheadAttention
import torch
from transformers.modeling_bert import BertConfig
from torch import nn
from torch.nn import functional as F
from collections import defaultdict


class AttentionPool(nn.Module):
    """attention pooling layer."""

    def __init__(self, hidden_size, drop=0.0):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.dropout = nn.Dropout(drop)

    def forward(self, input_, mask=None):
        """input: [B, T, D], mask = [B, T]"""
        score = self.fc(input_).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=input_.dtype) * -1e4
            score = score + mask
        norm_score = self.dropout(F.softmax(score, dim=1))
        output = norm_score.unsqueeze(1).matmul(input_).squeeze(1)
        return output


@VQA_MODELS.register_module()
class UniterForNlvr2PairedAttn(UniterBaseModel):
    """Finetune UNITER for NLVR2 (paired format with additional attention
    layer)"""

    def __init__(self, embeddings, encoder, **kwargs):
        super().__init__()
        self.uniter = UNITER(embeddings, encoder, **kwargs)
        config = BertConfig.from_json_file(encoder.config_file)
        self.config = config
        self.attn1 = MultiheadAttention(config.hidden_size, config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        self.attn2 = MultiheadAttention(config.hidden_size, config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        self.fc = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size), nn.ReLU(), nn.Dropout(config.hidden_dropout_prob))
        self.attn_pool = AttentionPool(config.hidden_size, config.attention_probs_dropout_prob)
        self.nlvr2_output = nn.Linear(2 * config.hidden_size, 2)
        self.apply(self.init_weights)
        self.init_type_embedding()
        self.uniter.load_my_state_dict(kwargs['pretrained_path'])
        self.set_dropout(kwargs['dropout'])

    def init_type_embedding(self):
        new_emb = nn.Embedding(3, self.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embedding[0].token_type_embeddings\
                .weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        new_emb.weight.data[2, :].copy_(emb)
        self.uniter.embedding[0].token_type_embeddings = new_emb

    def forward_train(self, data, output_all_encoded_layers=False, **kwargs):
        batch = defaultdict(lambda: None, {k: v.cuda() for k, v in data.items()})
        sequence_output = self.uniter(batch, output_all_encoded_layers=output_all_encoded_layers, **kwargs)

        # separate left image and right image
        bs, tl, d = sequence_output.size()
        left_out, right_out = sequence_output.contiguous().view(bs // 2, tl * 2, d).chunk(2, dim=1)
        # bidirectional attention
        mask = batch['attn_masks'] == 0
        left_mask, right_mask = mask.contiguous().view(bs // 2, tl * 2).chunk(2, dim=1)
        left_out = left_out.transpose(0, 1)
        right_out = right_out.transpose(0, 1)
        l2r_attn, _ = self.attn1(left_out, right_out, right_out, key_padding_mask=right_mask)
        r2l_attn, _ = self.attn2(right_out, left_out, left_out, key_padding_mask=left_mask)
        left_out = self.fc(torch.cat([l2r_attn, left_out], dim=-1)).transpose(0, 1)
        right_out = self.fc(torch.cat([r2l_attn, right_out], dim=-1)).transpose(0, 1)
        # attention pooling and final prediction
        left_out = self.attn_pool(left_out, left_mask)
        right_out = self.attn_pool(right_out, right_mask)
        answer_scores = self.nlvr2_output(torch.cat([left_out, right_out], dim=-1))

        try:
            model_outputs = {'scores': answer_scores, 'targets': batch['targets']}
        except Exception:
            # TODO ???
            pass
        return model_outputs
