import torch

from ..builder import VQA_MODELS, build_backbone, build_combine_layer, build_embedding, build_encoder, build_head
from .base_model import BaseModel


@VQA_MODELS.register_module()
class PYTHIA(BaseModel):

    def __init__(self, embedding, encoder, backbone, combine_model, head):
        super().__init__()

        self.embedding_model = build_embedding(embedding)
        self.encoder_model = build_encoder(encoder)
        self.backbone = build_backbone(backbone)
        self.combine_model = build_combine_layer(combine_model)  ###combine text and image
        self.head = build_head(head)  ###包括 classification head， generation head

        # self.init_weights()

    def process_text_embedding(self, text):

        # Get embedding models
        text_embedding_model = self.embedding_model[-1]
        embedding = text_embedding_model(text)
        text_embeddding_total = embedding

        return text_embeddding_total

    def process_feature_embedding(self, data, text_embedding_total, extra=None, batch_size_t=None):

        bs, num_feats, feats_dim = data['feature'].size()
        feature_embeddings = []
        feature_attentions = []
        features = []
        feature_global = data['feature_global'].reshape(bs, -1, feats_dim)[:, :num_feats, :]
        features.append(data['feature'].cuda())
        features.append(feature_global.cuda())
        bs_num_feats = num_feats * torch.ones(bs)

        # Now, iterate to get final attended image features
        for i, feature in enumerate(features):
            feature_encoder = self.encoder_model[i]

            # Encode the features
            encoded_feature = feature_encoder(feature)

            feature_embedding_model = self.backbone[i]

            # Forward through these embeddings one by one
            # for feature_embedding_model in feature_embedding_models:
            inp = (encoded_feature, text_embedding_total, bs_num_feats, extra)

            embedding, attention = feature_embedding_model(*inp)
            feature_embeddings.append(embedding)
            feature_attentions.append(attention.squeeze(-1))

        # Concatenate all features embeddings and return along with attention
        feature_embedding_total = torch.cat(feature_embeddings, dim=1)
        return feature_embedding_total, feature_attentions

    def process_context_feature_embedding(self, data, text_embedding_total, order_vectors=None, batch_size_t=None):

        feature_embeddings = []
        feature_attentions = []
        features = []

        features = data['context_feature1']
        feature_dim = data['context_feature1'].shape[1] * torch.ones(data['context_feature1'].shape[0])
        # Now, iterate to get final attended image features

        feature_encoder = self.encoder_model[-1]
        # Encode the features
        encoded_feature = feature_encoder(features)
        feature_embedding_model = self.backbone[-1]
        # Forward through these embeddings one by one
        # for feature_embedding_model in feature_embedding_models:
        inp = (encoded_feature, text_embedding_total, feature_dim, order_vectors)
        embedding, attention = feature_embedding_model(*inp)
        feature_embeddings.append(embedding)
        feature_attentions.append(attention.squeeze(-1))

        # Concatenate all features embeddings and return along with attention
        feature_embedding_total = torch.cat(feature_embeddings, dim=1)
        return feature_embedding_total, feature_attentions

    def forward_train(self, data):
        text = self.embedding_model[0](data['input_ids'].cuda())
        text_embedding_total = self.process_text_embedding(text)

        image_embedding_total, _ = self.process_feature_embedding(data, text_embedding_total)

        joint_embedding = self.combine_model(image_embedding_total, text_embedding_total)
        targets = data['answers_scores'].cuda()

        scores = self.head(joint_embedding)

        model_output = {'scores': scores, 'target': targets}

        return model_output

    def forward_test(self, data):
        model_output = self.forward_train(data)
        return model_output
