from imix.models.builder import VQA_MODELS
from .unitervqa import UNITERVQA


@VQA_MODELS.register_module()
class UNITERVE(UNITERVQA):

    def __init__(self, embeddings, encoder, head, **kwargs):
        # out_dim: entailment\ neutral\ contradiction
        assert head['out_dim'] == 3
        super().__init__(embeddings, encoder, head, **kwargs)
        self.uniter.load_my_state_dict(kwargs['pretrained_path'])
        self.set_dropout(kwargs['dropout'])
