from ..builder import VQA_MODELS
from .base_model import BaseModel
from ..devlbert import DeVLBertForVLTasks, BertConfig


@VQA_MODELS.register_module()
class DeVLBert(BaseModel):

    def __init__(self, config):
        super().__init__()
        self.model = DeVLBertForVLTasks.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            config=BertConfig.from_json_file(config.bert_file_path),
            num_labels=config.num_labels,
            default_gpu=config.default_gpu)

    def preprocess_data(self, data):
        question = data[3].cuda()
        features = data[0].cuda()
        spatials = data[1].cuda()
        segment_ids = data[6].cuda()
        input_mask = data[5].cuda()
        image_mask = data[2].cuda()
        co_attention_mask = data[7].cuda()
        target = data[4].cuda()
        (vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, vision_logit, linguisic_prediction,
         linguisic_logit) = self.model(question, features, spatials, segment_ids, input_mask, image_mask,
                                       co_attention_mask)
        model_output = {'scores': vil_prediction, 'target': target}
        return model_output

    def forward_train(self, data, **kwargs):
        self.model.train()
        return self.preprocess_data(data)

    def forward_test(self, data, **kwargs):
        self.model.eval()
        return self.preprocess_data(data)
