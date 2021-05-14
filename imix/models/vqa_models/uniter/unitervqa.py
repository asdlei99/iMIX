from imix.models.builder import VQA_MODELS, build_head
from imix.models.vqa_models.base_model import UniterBaseModel
from .uniter import UNITER
from collections import defaultdict


@VQA_MODELS.register_module()
class UNITERVQA(UniterBaseModel):

    def __init__(self, embeddings, encoder, head, **kwargs):
        super().__init__()
        self.uniter = UNITER(embeddings, encoder, **kwargs)
        self.head = build_head(head)

        self.uniter.load_my_state_dict(kwargs['pretrained_path'])
        self.set_dropout(kwargs['dropout'])

    def forward_train(self, data, output_all_encoded_layers=False, **kwargs):
        batch = defaultdict(lambda: None, {k: v.cuda() for k, v in data.items()})
        encoded_output = self.uniter(batch, output_all_encoded_layers=output_all_encoded_layers, **kwargs)
        pooled_output = self.uniter.encoder.pooler(encoded_output)
        logits = self.head(pooled_output)
        try:
            model_outputs = {'scores': logits, 'target': batch['targets']}
        except Exception:
            # TODO ???
            pass
        return model_outputs

    def forward_test(self, data, **kwargs):
        model_output = self.forward_train(data)
        return model_output
