import torch
from .base_model import BaseModel
from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder, build_head, build_combine_layer
from ..encoder import OSCARBackbone



@VQA_MODELS.register_module()
class UNITER(BaseModel):

  def __init__(self, embeddings, encoder, head, **kwargs):
    super().__init__()

    self.embedding = build_embedding(embeddings)
    self.encoder = build_encoder(encoder)
    self.head = build_head(head)
    pretrained_path = kwargs['pretrained_path']
    ckpt = torch.load(pretrained_path)
    self.load_my_state_dict(ckpt)

  #### not load head weights from pretrained weights
  def load_my_state_dict(self, state_dict):
      own_state = self.state_dict()
      for name, param in state_dict.items():
          if 'uniter.embeddings' in name:
              own_state[name.replace('uniter.embeddings', 'embedding.0')].copy_(param)
          elif 'uniter.img_embeddings' in name:
              own_state[name.replace('uniter.img_embeddings', 'embedding.1')].copy_(param)
          elif 'uniter.encoder' in name:
              own_state[name.replace('uniter.encoder', 'encoder')].copy_(param)
          elif 'uniter.pooler' in name:
              own_state[name.replace('uniter.pooler', 'encoder.pooler')].copy_(param)
          if name not in own_state:
              continue
          own_state[name].copy_(param)

  def _compute_txt_embeddings(self, input_ids, position_ids, txt_type_ids=None):
    output = self.embedding[0](input_ids, position_ids, txt_type_ids)
    return output

  def _compute_img_embeddings(self,
                              img_feat,
                              img_pos_feat,
                              img_masks=None,
                              img_type_ids=None):
    if img_type_ids is None:
      img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
    img_type_embeddings = self.embedding[0].token_type_embeddings(img_type_ids)
    output = self.embedding[1](img_feat, img_pos_feat, img_type_embeddings,
                               img_masks)
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
    txt_emb = self._compute_txt_embeddings(input_ids, position_ids,
                                           txt_type_ids)
    img_emb = self._compute_img_embeddings(img_feat, img_pos_feat, img_masks,
                                           img_type_ids)
    if gather_index is not None:
      # align back to most compact input
      gather_index = gather_index.unsqueeze(-1).expand(-1, -1,
                                                       self.config.hidden_size)
      embedding_output = torch.gather(
          torch.cat([txt_emb, img_emb], dim=1), dim=1, index=gather_index)
    else:
      embedding_output = torch.cat([txt_emb, img_emb], dim=1)

    return embedding_output

  def forward(self, data, output_all_encoded_layers=False):

    input_ids = data['input_ids'].cuda()
    position_ids = torch.arange(
        0, input_ids.size(1), dtype=torch.long).unsqueeze(0).cuda()

    img_feat = data['feature'].cuda()
    img_pos_feat = data['bbox'].cuda()

    image_mask = (torch.arange(img_feat.size(-2)).expand(*img_feat.size()[:-1]))
    if len(data['image_dim'].size()) < len(image_mask.size()):
      data['image_dim'] = data['image_dim'].unsqueeze(-1)
    image_mask = image_mask < data['image_dim']

    txt_type_ids = None
    img_type_ids = None
    img_masks = None
    image_mask = image_mask.long()
    # gather_index = data.gather_index
    attention_mask = torch.cat((data['input_mask'], image_mask), dim=1).cuda()

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(
        dtype=next(self.parameters()).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    embedding_output = self._compute_img_txt_embeddings(input_ids, position_ids,
                                                        img_feat, img_pos_feat,
                                                        None, img_masks,
                                                        txt_type_ids,
                                                        img_type_ids)

    encoded_layers = self.encoder(
        embedding_output,
        extended_attention_mask,
        output_all_encoded_layers=output_all_encoded_layers)
    if not output_all_encoded_layers:
      encoded_layers = encoded_layers[-1]

    pooled_output = self.encoder.pooler(encoded_layers)
    logits = self.head(pooled_output)
    model_outputs={
        'scores':logits,
        'target':data['answers_scores'].cuda()
    }
    return model_outputs
