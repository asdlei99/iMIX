import torch.nn as nn
import torch
from ..builder import ENCODER
import os
import logging
from copy import deepcopy
import pickle
import math
from transformers.modeling_bert import (BertConfig, BertEmbeddings, BertEncoder, BertPreTrainedModel,
                                        BertPredictionHeadTransform, BertPooler, BertLayer, BertOutput,
                                        BertIntermediate, ACT2FN)
from imix.models.embedding import BertVisioLinguisticEmbeddings, BertImageFeatureEmbeddings

logger = logging.getLogger(__name__)

TEXT_BERT_HIDDEN_SIZE = 768


class BertImagePooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertTextPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImageSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention '
                             'heads (%d)' % (config.v_hidden_size, config.v_num_attention_heads))
        self.dynamic_attention = config.dynamic_attention
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(config.v_hidden_size / config.v_num_attention_heads)

        self.visualization = config.visualization

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)

        if self.dynamic_attention:
            self.dyLinear_q = nn.Linear(config.hidden_size, self.all_head_size)
            self.dyLinear_k = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if self.dynamic_attention:
            pool_embedding = (txt_embedding * txt_attention_mask).sum(1)
            pool_embedding = pool_embedding / txt_attention_mask.sum(1)

            # given pool embedding, Linear and Sigmoid layer.
            gate_q = 1 + torch.sigmoid(self.dyLinear_q(pool_embedding))
            gate_k = 1 + torch.sigmoid(self.dyLinear_k(pool_embedding))

            mixed_query_layer = mixed_query_layer * gate_q.unsqueeze(1)
            mixed_key_layer = mixed_key_layer * gate_k.unsqueeze(1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the
        # raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel
        # forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.visualization:
            attn_data = {
                'attn': attention_probs,
                'queries': query_layer,
                'keys': key_layer,
            }
        else:
            attn_data = None

        return context_layer, attn_data


class BertImageSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask, txt_embedding, txt_attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask, txt_embedding, txt_attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertImageIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_intermediate_size)
        if isinstance(config.v_hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_intermediate_size, config.v_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask, txt_embedding,
                                                           txt_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertBiAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention '
                             'heads (%d)' % (config.bi_hidden_size, config.bi_num_attention_heads))

        self.visualization = config.visualization
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(config.bi_hidden_size / config.bi_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)

        self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw
        # attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1
        # if use_co_attention_mask:
        # attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size, )
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the
        # raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel
        # forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2 + attention_mask2
        # if use_co_attention_mask:
        # attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size, )
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        attn_data = None

        if self.visualization:
            attn_data = {
                'attn1': attention_probs1,
                'queries1': query_layer2,
                'keys1': key_layer1,
                'attn2': attention_probs2,
                'querues2': query_layer1,
                'keys2': key_layer2,
            }

        return context_layer1, context_layer2, attn_data


class BertBiOutput(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.q_dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.q_dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.q_dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.q_dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.biattention = BertBiAttention(config)

        self.biOutput = BertBiOutput(config)

        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)

        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask,
            use_co_attention_mask,
        )

        attention_output1, attention_output2 = self.biOutput(bi_output2, input_tensor1, bi_output1, input_tensor2)

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.

        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id
        self.t_biattention_id = config.t_biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer
        layer = BertLayer(config)
        v_layer = BertImageLayer(config)
        connect_layer = BertConnectionLayer(config)

        self.layer = nn.ModuleList([deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.v_layer = nn.ModuleList([deepcopy(v_layer) for _ in range(config.v_num_hidden_layers)])
        self.c_layer = nn.ModuleList([deepcopy(connect_layer) for _ in range(len(config.v_biattention_id))])

    def forward(
        self,
        txt_embedding,
        image_embedding,
        txt_attention_mask,
        txt_attention_mask2,
        image_attention_mask,
        co_attention_mask=None,
        output_all_encoded_layers=True,
        output_all_attention_masks=False,
    ):

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []

        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_c = []

        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = image_embedding.size()

        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.v_biattention_id, self.t_biattention_id):

            v_end = v_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end

            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    txt_embedding, txt_attention_probs = self.layer[idx](txt_embedding, txt_attention_mask)
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks:
                        all_attention_mask_t.append(txt_attention_probs)

            for idx in range(t_start, t_end):
                txt_embedding = self.layer[idx](txt_embedding, txt_attention_mask)
                txt_embedding = txt_embedding[0]
                # txt_embedding, txt_attention_probs = self.layer[idx](
                #     txt_embedding, txt_attention_mask
                # )
                # if output_all_attention_masks:
                #     all_attention_mask_t.append(txt_attention_probs)

            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    image_embedding, image_attention_probs = self.v_layer[idx](
                        image_embedding,
                        image_attention_mask,
                        txt_embedding,
                        txt_attention_mask2,
                    )
                    v_start = self.fixed_v_layer

                    if output_all_attention_masks:
                        all_attnetion_mask_v.append(image_attention_probs)

            for idx in range(v_start, v_end):
                image_embedding, image_attention_probs = self.v_layer[idx](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask2,
                )

                if output_all_attention_masks:
                    all_attnetion_mask_v.append(image_attention_probs)

            if count == 0 and self.in_batch_pairs:
                # new batch size is the batch_size ^2
                image_embedding = (
                    image_embedding.unsqueeze(0).expand(batch_size, batch_size, num_regions,
                                                        v_hidden_size).contiguous().view(
                                                            batch_size * batch_size, num_regions, v_hidden_size))
                image_attention_mask = (
                    image_attention_mask.unsqueeze(0).expand(batch_size, batch_size, 1, 1,
                                                             num_regions).contiguous().view(
                                                                 batch_size * batch_size, 1, 1, num_regions))

                txt_embedding = (
                    txt_embedding.unsqueeze(1).expand(batch_size, batch_size, num_words,
                                                      t_hidden_size).contiguous().view(
                                                          batch_size * batch_size, num_words, t_hidden_size))
                txt_attention_mask = (
                    txt_attention_mask.unsqueeze(1).expand(batch_size, batch_size, 1, 1, num_words).contiguous().view(
                        batch_size * batch_size, 1, 1, num_words))
                co_attention_mask = (
                    co_attention_mask.unsqueeze(1).expand(batch_size, batch_size, 1, num_regions,
                                                          num_words).contiguous().view(
                                                              batch_size * batch_size, 1, num_regions, num_words))

            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(
                    image_embedding.size(0),
                    txt_embedding.size(1),
                    txt_embedding.size(2),
                )
                txt_attention_mask = txt_attention_mask.expand(
                    image_embedding.size(0),
                    txt_attention_mask.size(1),
                    txt_attention_mask.size(2),
                    txt_attention_mask.size(3),
                )

            if self.with_coattention:
                # do the bi attention.
                image_embedding, txt_embedding, co_attention_probs = self.c_layer[count](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask,
                    co_attention_mask,
                    use_co_attention_mask,
                )

                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)

            v_start = v_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)

        for idx in range(v_start, len(self.v_layer)):
            image_embedding, image_attention_probs = self.v_layer[idx](
                image_embedding,
                image_attention_mask,
                txt_embedding,
                txt_attention_mask2,
            )

            if output_all_attention_masks:
                all_attnetion_mask_v.append(image_attention_probs)

        for idx in range(t_start, len(self.layer)):
            txt_embedding = self.layer[idx](txt_embedding, txt_attention_mask)

            # txt_embedding, txt_attention_probs = self.layer[idx](
            #     txt_embedding, txt_attention_mask
            # )
            # if output_all_attention_masks:
            #     all_attention_mask_t.append(txt_attention_probs)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)

        return (
            all_encoder_layers_t,
            all_encoder_layers_v,
            (all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c),
        )


@ENCODER.register_module()
class ViLBERTBase(BertPreTrainedModel):

    def __init__(
        self,
        config,
        # visual_embedding_dim=512,
        # embedding_strategy="plain",
        # bypass_transformer=False,
        # output_attentions=False,
        # output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        # config.visual_embedding_dim = visual_embedding_dim
        # config.embedding_strategy = embedding_strategy
        # config.bypass_transformer = bypass_transformer
        # config.output_attentions = output_attentions
        # config.output_hidden_states = output_hidden_states

        # initilize word embedding
        self.embeddings = BertEmbeddings(config)

        self.task_specific_tokens = config.task_specific_tokens
        self.v_embeddings = BertImageFeatureEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)

        self.init_weights()

    def forward(
        self,
        input_txt,
        image_feature,
        image_location,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        task_ids=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(image_feature.size(0), image_feature.size(1)).type_as(input_txt)

        if self.task_specific_tokens:
            # extend the mask
            mask_tokens = input_txt.new().resize_(input_txt.size(0), 1).fill_(1)
            attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask2 = attention_mask.unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask2 = extended_attention_mask2.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility

        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility

        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(input_txt.size(0), image_feature.size(1),
                                            input_txt.size(1)).type_as(extended_image_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)

        # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility

        embedding_output = self.embeddings(input_txt, token_type_ids, task_ids)
        v_embedding_output = self.v_embeddings(image_feature, image_location)
        encoded_layers_t, encoded_layers_v, all_attention_mask = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_attention_mask2,
            extended_image_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]

        pooled_output_t = self.t_pooler(sequence_output_t[0])
        pooled_output_v = self.v_pooler(sequence_output_v)

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]

        return (
            encoded_layers_t,
            encoded_layers_v,
            pooled_output_t,
            pooled_output_v,
            all_attention_mask,
        )


@ENCODER.register_module()
class ViLBERTForPretraining(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self.output_attentions = self.config['output_attentions']
        self.output_hidden_states = self.config['output_hidden_states']

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = getattr(self.config, 'bert_model_name', None)
        self.bert_config = BertConfig.from_dict(OmegaConf.to_container(self.config, resolve=True))
        if self.bert_model_name is None:
            self.bert = ViLBERTBase(
                self.bert_config,
                # visual_embedding_dim=self.config.visual_embedding_dim,
                # embedding_strategy=self.config.embedding_strategy,
                # bypass_transformer=self.config.bypass_transformer,
                # output_attentions=self.config.output_attentions,
                # output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = ViLBERTBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(get_mmf_cache_dir(), 'distributed_{}'.format(-1)),
                # visual_embedding_dim=self.config.visual_embedding_dim,
                # embedding_strategy=self.config.embedding_strategy,
                # bypass_transformer=self.config.bypass_transformer,
                # output_attentions=self.config.output_attentions,
                # output_hidden_states=self.config.output_hidden_states,
            )

        self.vocab_size = self.bert.config.vocab_size

        # TODO: Once omegaconf fixes int keys issue, bring this back
        # See https://github.com/omry/omegaconf/issues/149
        # with omegaconf.open_dict(self.config):
        #     # Add bert config such as hidden_state to our main config
        #     self.config.update(self.bert.config.to_dict())
        if self.bert_model_name is None:
            bert_masked_lm = BertForPreTraining(self.bert.config)
        else:
            bert_masked_lm = BertForPreTraining.from_pretrained(
                self.config.bert_model_name,
                cache_dir=os.path.join(get_mmf_cache_dir(), 'distributed_{}'.format(-1)),
            )
        self.cls = deepcopy(bert_masked_lm.cls)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)

            self.tie_weights()

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.

        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self.bert._tie_or_clone_weights(self.cls.predictions.decoder, self.bert.embeddings.word_embeddings)

    def forward(
        self,
        input_ids,
        input_mask,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        masked_lm_labels=None,
    ):
        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            position_embeddings_visual,
            visual_embeddings_type,
            image_text_alignment,
        )

        output_dict = {}

        if self.output_attentions:
            output_dict['attention_weights'] = attention_weights

        if self.output_hidden_states:
            output_dict['sequence_output'] = sequence_output
            output_dict['pooled_output'] = pooled_output

        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        if masked_lm_labels is not None:
            output_dict['logits'] = prediction_scores
            masked_lm_loss = self.loss_fct(
                prediction_scores.contiguous().view(-1, self.vocab_size),
                masked_lm_labels.contiguous().view(-1),
            )
            output_dict['masked_lm_loss'] = masked_lm_loss
            output_dict['loss'] = masked_lm_loss

        return output_dict


@ENCODER.register_module()
class ViLBERTForClassification(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        # self.output_attentions = self.config["output_attentions"]
        # self.output_hidden_states = self.config["output_hidden_states"]
        # self.pooler_strategy = self.config.get("pooler_strategy", "default")

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = self.config['bert_model_name']
        self.bert_config = BertConfig.from_dict(self.config)
        if self.bert_model_name is None:
            self.bert = ViLBERTBase(
                self.bert_config,
                # visual_embedding_dim=self.config["visual_embedding_dim"],
                # embedding_strategy=self.config["embedding_strategy"],
                # bypass_transformer=self.config["bypass_transformer"],
                # output_attentions=self.config["output_attentions"],
                # output_hidden_states=self.config["output_hidden_states"],
            )
        else:
            self.bert = ViLBERTBase.from_pretrained(
                self.config['bert_model_name'],
                config=self.bert_config,
                cache_dir=os.path.join('/home/jinliang/.cache/torch', 'transformers'.format(-1)),
                # visual_embedding_dim=self.config["visual_embedding_dim"],
                # embedding_strategy=self.config["embedding_strategy"],
                # bypass_transformer=self.config["bypass_transformer"],
                # output_attentions=self.config["output_attentions"],
                # output_hidden_states=self.config["output_hidden_states"],
            )

        self.training_head_type = self.config['training_head_type']
        self.num_labels = self.config['num_labels']
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        if self.config['training_head_type'] == 'nlvr2':
            self.bert.config.hidden_size *= 2
        self.bert.config.hidden_size = self.config['bi_hidden_size']
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.bert.config),
            nn.Linear(self.bert.config.hidden_size, self.config['num_labels']),
        )

        self.init_weights()

    def init_weights(self):
        if self.config['random_initialize'] is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids,
        image_feature,
        image_location,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        masked_lm_labels=None,
        image_label=None,
        image_target=None,
        next_sentence_label=None,
        output_all_attention_masks=False,
    ):

        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            attention_weights,
        ) = self.bert(
            input_ids,
            image_feature,
            image_location,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks,
        )

        output = {}
        if output_all_attention_masks:
            output['attention_weights'] = attention_weights

        if self.config['fusion_method'] == 'sum':
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.config['fusion_method'] == 'mul':
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise AssertionError

        if self.training_head_type == 'nlvr2':
            pooled_output = pooled_output.view(-1, pooled_output.size(1) * 2)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output['scores'] = reshaped_logits

        return output
