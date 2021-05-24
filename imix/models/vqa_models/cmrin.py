from pytorch_pretrained_bert.modeling import BertModel

from ..builder import VQA_MODELS, build_backbone, build_encoder
from .base_model import BaseModel


@VQA_MODELS.register_module()
class CMRIN(BaseModel):

    def __init__(self, encoder, backbone, weights_file):
        super().__init__()

        self.encoder_model = build_encoder(encoder)
        self.encoder_model.load_weights(weights_file)
        self.textmodel = BertModel.from_pretrained('bert-base-uncased')
        self.backbone = build_backbone(backbone)

    def forward_train(self, data):
        input_mask = data['input_mask'].cuda()
        image = data['image'].cuda()
        input_ids = data['input_ids'].cuda()

        raw_fvisu = self.encoder_model(image)

        # Language Module
        all_encoder_layers, _ = self.textmodel(input_ids, token_type_ids=None, attention_mask=input_mask)
        # Sentence feature at the first position [cls]
        raw_flang = (all_encoder_layers[-1][:, 0, :] + all_encoder_layers[-2][:, 0, :] +
                     all_encoder_layers[-3][:, 0, :] + all_encoder_layers[-4][:, 0, :]) / 4
        raw_flang = raw_flang.detach()

        pred_anchor = self.backbone(raw_flang, raw_fvisu)

        model_output = {'scores': pred_anchor, 'target': data['bbox'].cuda()}
        return model_output

    def forward_test(self, data):
        model_output = self.forward_train(data)
        return model_output
