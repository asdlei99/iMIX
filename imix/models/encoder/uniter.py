import torch.nn as nn
from transformers.modeling_bert import BertLayer, BertConfig
import copy

import torch
from ..builder import ENCODER


@ENCODER.register_module()
class UniterEncoder(nn.Module):

  def __init__(self, config_file):
    super().__init__()
    config = BertConfig.from_json_file(config_file)
    layer = BertLayer(config)
    self.layer = nn.ModuleList(
        [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
    self.pooler = BertPooler(config)

  def forward(self, input_, attention_mask, output_all_encoded_layers=True):
    all_encoder_layers = []
    hidden_states = input_
    for layer_module in self.layer:
      hidden_states = layer_module(hidden_states, attention_mask)
      hidden_states = hidden_states[0]
      if output_all_encoded_layers:
        all_encoder_layers.append(hidden_states)
    if not output_all_encoded_layers:
      all_encoder_layers.append(hidden_states)
    return all_encoder_layers

class BertPooler(nn.Module):
  def __init__(self, config):
    super(BertPooler, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.activation = nn.Tanh()
  def forward(self, hidden_states):
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.
    first_token_tensor = hidden_states[:, 0]
    pooled_output = self.dense(first_token_tensor)
    pooled_output = self.activation(pooled_output)
    return pooled_output