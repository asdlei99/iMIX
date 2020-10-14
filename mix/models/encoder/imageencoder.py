import torch.nn as nn
import torch
from ..builder import ENCODER
import os
import pickle


@ENCODER.register_module()
class ImageFeatureEncoder(nn.Module):

    def __init__(self,
                 encoder_type,
                 in_dim=2048,
                 weights_file=None,
                 bias_file=None,
                 **kwargs):
        super().__init__()

        if encoder_type == 'default':
            self.module = Identity()
            self.module.in_dim = in_dim
            self.module.out_dim = in_dim
        elif encoder_type == 'projection':
            module_type = kwargs.pop('module', 'linear')
            self.module = ProjectionEmbedding(module_type, in_dim, **kwargs)
        elif encoder_type == 'finetune_faster_rcnn_fpn_fc7':
            self.module = FinetuneFasterRcnnFpnFc7(in_dim, weights_file,
                                                   bias_file, **kwargs)
        else:
            raise NotImplementedError('Unknown Image Encoder: %s' %
                                      encoder_type)

        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class FinetuneFasterRcnnFpnFc7(nn.Module):

    def __init__(self, in_dim, weights_file, bias_file, *args, **kwargs):
        super().__init__()

        # model_data_dir = get_absolute_path(model_data_dir)
        #
        # if not os.path.isabs(weights_file):
        #     weights_file = os.path.join(model_data_dir, weights_file)
        # if not os.path.isabs(bias_file):
        #     bias_file = os.path.join(model_data_dir, bias_file)

        if not os.path.exists(bias_file) or not os.path.exists(weights_file):
            download_path = download_pretrained_model('detectron.vmb_weights')
            weights_file = get_absolute_path(
                os.path.join(download_path, 'fc7_w.pkl'))
            bias_file = get_absolute_path(
                os.path.join(download_path, 'fc7_b.pkl'))

        with open(weights_file, 'rb') as w:
            weights = pickle.load(w)
        with open(bias_file, 'rb') as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3
