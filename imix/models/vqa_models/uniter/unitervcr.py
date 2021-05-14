from imix.models.builder import VQA_MODELS, build_head
from imix.models.vqa_models.base_model import UniterBaseModel
from .uniter import UNITER
from torch import nn
from collections import defaultdict
from transformers.modeling_bert import BertConfig

NUM_SPECIAL_TOKENS = 81


@VQA_MODELS.register_module()
class UNITERVCR(UniterBaseModel):

    def __init__(self, embeddings, encoder, head, **kwargs):
        super().__init__()
        config = BertConfig.from_json_file(encoder.config_file)
        self.config = config
        self.uniter = UNITER(embeddings, encoder, **kwargs)
        self.head = build_head(head)
        self.init_type_embedding()
        self.init_word_embedding(NUM_SPECIAL_TOKENS)
        self.uniter.load_my_state_dict(kwargs['pretrained_path'])
        self.set_dropout(kwargs['dropout'])

    def forward_train(self, data, output_all_encoded_layers=False, **kwargs):
        batch = defaultdict(lambda: None, {k: v.cuda() for k, v in data.items()})
        encoded_output = self.uniter(batch, output_all_encoded_layers=output_all_encoded_layers, **kwargs)
        pooled_output = self.uniter.encoder.pooler(encoded_output)
        logits = self.head(pooled_output)
        try:
            # TODO, confirm 'targets' key word in dataloader
            model_outputs = {'scores': logits, 'targets': batch['targets'].squeeze(-1)}
        except Exception:
            pass
        return model_outputs

    def forward_test(self, data, **kwargs):
        model_output = self.forward_train(data)
        return model_output

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embedding[0].token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embedding[0].token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embedding[0].token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embedding[0].word_embeddings.weight.size(0)
        new_emb = nn.Embedding(orig_word_num + num_special_tokens, self.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embedding[0].word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embedding[0].word_embeddings = new_emb
