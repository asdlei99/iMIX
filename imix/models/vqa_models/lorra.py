from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder, build_head, build_combine_layer
import torch.nn as nn
import torch
from .pythia import PYTHIA


@VQA_MODELS.register_module()
class LoRRA(PYTHIA):

  def __init__(self, **kwargs):
    super(LoRRA, self).__init__(**kwargs)
    # self  bedding_model = build_embedding(embedding)
    # self.encoder_model = build_encoder(encoder)
    # self.backbone = build_backbone(backbone)
    # self.combine_model = build_combine_layer(combine_model)  ###combine text and image
    # self.head = build_head(head)  ###包括 classification head， generation head

    # self.init_weights()

  # def process_text_embedding(self, text):
  #
  #     # Get embedding models
  #     text_embedding_model = self.embedding_model[-1]
  #     embedding = text_embedding_model(text)
  #     text_embeddding_total = embedding
  #
  #     return text_embeddding_total
  #
  # def process_feature_embedding(self, data, text_embedding_total, extra=None, batch_size_t=None):
  #
  #     feature_embeddings = []
  #     feature_attentions = []
  #     features = []
  #
  #     # batch_size_t = (data.get_batch_size() if batch_size_t is None else batch_size_t)
  #     # feature_idx = 0
  #
  #     # 得到两种图像特征,ROI 和 GRID
  #     # while True:
  #     #     feature = getattr(data, f"image_feature_{feature_idx:d}", None)
  #     #     if feature is None:
  #     #         break
  #     #     feature_idx += 1
  #     #     feature = feature[:batch_size_t]
  #     #     features.append(feature)
  #
  #     features.append(data['img_feature1'])
  #     features.append(data['img_feature2'])
  #     feature_dim = data['img_feature1'].shape[1] * torch.ones(data['img_feature1'].shape[0])
  #     # Now, iterate to get final attended image features
  #     for i, feature in enumerate(features):
  #         feature_encoder = self.encoder_model[i]
  #
  #         # Encode the features
  #         encoded_feature = feature_encoder(feature)
  #
  #         feature_embedding_model = self.backbone[i]
  #
  #         # Forward through these embeddings one by one
  #         # for feature_embedding_model in feature_embedding_models:
  #         inp = (encoded_feature, text_embedding_total, feature_dim, extra)
  #
  #         embedding, attention = feature_embedding_model(*inp)
  #         feature_embeddings.append(embedding)
  #         feature_attentions.append(attention.squeeze(-1))
  #
  #     # Concatenate all features embeddings and return along with attention
  #     feature_embedding_total = torch.cat(feature_embeddings, dim=1)
  #     return feature_embedding_total, feature_attentions

  def forward_train(self, data):
    text = self.embedding_model[0](data['input_ids'].cuda())
    text_embedding_total = self.process_text_embedding(text)

    image_embedding_total, _ = self.process_feature_embedding(
        data, text_embedding_total)

    context_embedding_total, _ = self.process_context_feature_embedding(
        data, text_embedding_total, data['order_vectors'])

    joint_embedding = self.combine_model(image_embedding_total,
                                         text_embedding_total,
                                         context_embedding_total)

    model_output = {'scores': self.head(joint_embedding)}

    return model_output

  def forward_test(self, data):
    model_output = self.forward_train(data)
    return model_output