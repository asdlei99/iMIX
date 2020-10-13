import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm

from ..builder import BACKBONES
from ..combine_layers import ModalCombineLayer


class Linear(nn.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # compatible with xavier_initializer in TensorFlow
        fan_avg = (self.in_features + self.out_features) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)


activations = {
    'NON': lambda x: x,
    'TANH': torch.tanh,
    'SIGMOID': F.sigmoid,
    'RELU': F.relu,
    'ELU': F.elu,
}


def apply_mask1d(attention, image_locs):
    batch_size, num_loc = attention.size()
    tmp1 = attention.new_zeros(num_loc)
    tmp1[:num_loc] = torch.arange(
        0, num_loc, dtype=attention.dtype).unsqueeze(0)

    tmp1 = tmp1.expand(batch_size, num_loc)
    tmp2 = image_locs.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask = torch.ge(tmp1, tmp2)
    attention = attention.masked_fill(mask, -1e30)
    return attention


def apply_mask2d(attention, image_locs):
    batch_size, num_loc, _ = attention.size()
    tmp1 = attention.new_zeros(num_loc)
    tmp1[:num_loc] = torch.arange(
        0, num_loc, dtype=attention.dtype).unsqueeze(0)

    tmp1 = tmp1.expand(batch_size, num_loc)
    tmp2 = image_locs.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask1d = torch.ge(tmp1, tmp2)
    mask2d = mask1d[:, None, :] | mask1d[:, :, None]
    attention = attention.masked_fill(mask2d, -1e30)
    return attention


def generate_scaled_var_drop_mask(shape, keep_prob):
    assert keep_prob > 0. and keep_prob <= 1.
    mask = torch.rand(shape, device='cpu').le(keep_prob)  ##cuda
    mask = mask.float() / keep_prob
    return mask


@BACKBONES.register_module()
class LCGN_BACKBONE(nn.Module):

    def __init__(self, stem_linear, D_FEAT, CTX_DIM, CMD_DIM, MSG_ITER_NUM,
                 stemDropout, readDropout, memoryDropout, CMD_INPUT_ACT,
                 STEM_NORMALIZE):
        super().__init__()

        self.STEM_LINEAR = stem_linear
        self.D_FEAT = D_FEAT
        self.CTX_DIM = CTX_DIM
        self.CMD_DIM = CMD_DIM
        self.stemDropout = stemDropout
        self.readDropout = readDropout
        self.memoryDropout = memoryDropout

        self.MSG_ITER_NUM = MSG_ITER_NUM
        self.CMD_INPUT_ACT = CMD_INPUT_ACT
        self.STEM_NORMALIZE = STEM_NORMALIZE

        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_propagate_message()

    def build_loc_ctx_init(self):
        assert self.STEM_LINEAR == True
        if self.STEM_LINEAR:
            self.initKB = Linear(self.D_FEAT, self.CTX_DIM)
            self.x_loc_drop = nn.Dropout(1 - self.stemDropout)

        self.initMem = nn.Parameter(torch.randn(1, 1, self.CTX_DIM))

    def build_extract_textual_command(self):
        self.qInput = Linear(self.CMD_DIM, self.CMD_DIM)
        for t in range(self.MSG_ITER_NUM):
            qInput_layer2 = Linear(self.CMD_DIM, self.CMD_DIM)
            setattr(self, 'qInput%d' % t, qInput_layer2)
        self.cmd_inter2logits = Linear(self.CMD_DIM, 1)

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - self.readDropout)
        self.project_x_loc = Linear(self.CTX_DIM, self.CTX_DIM)
        self.project_x_ctx = Linear(self.CTX_DIM, self.CTX_DIM)
        self.queries = Linear(3 * self.CTX_DIM, self.CTX_DIM)
        self.keys = Linear(3 * self.CTX_DIM, self.CTX_DIM)
        self.vals = Linear(3 * self.CTX_DIM, self.CTX_DIM)
        self.proj_keys = Linear(self.CMD_DIM, self.CTX_DIM)
        self.proj_vals = Linear(self.CMD_DIM, self.CTX_DIM)
        self.mem_update = Linear(2 * self.CTX_DIM, self.CTX_DIM)
        self.combine_kb = Linear(2 * self.CTX_DIM, self.CTX_DIM)

    def forward(self, images, q_encoding, lstm_outputs, q_length, entity_num):
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(images)
        for t in range(self.MSG_ITER_NUM):
            x_ctx = self.run_message_passing_iter(q_encoding, lstm_outputs,
                                                  q_length, x_loc, x_ctx,
                                                  x_ctx_var_drop, entity_num,
                                                  t)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))
        return x_out

    def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
        qInput_layer2 = getattr(self, 'qInput%d' % t)
        act_fun = activations[self.CMD_INPUT_ACT]
        q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding)))
        raw_att = self.cmd_inter2logits(q_cmd[:, None, :] *
                                        lstm_outputs).squeeze(-1)
        raw_att = apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat([x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint)
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(self.CTX_DIM))
        edge_score = apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new

    def run_message_passing_iter(self, q_encoding, lstm_outputs, q_length,
                                 x_loc, x_ctx, x_ctx_var_drop, entity_num, t):
        cmd = self.extract_textual_command(q_encoding, lstm_outputs, q_length,
                                           t)
        x_ctx = self.propagate_message(cmd, x_loc, x_ctx, x_ctx_var_drop,
                                       entity_num)
        return x_ctx

    def loc_ctx_init(self, images):
        if self.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
        if self.STEM_LINEAR:
            # print(self.initKB.state_dict()['weight'].size()[-1])
            # print(images.size()[-1])
            x_loc = self.initKB(images)

            x_loc = self.x_loc_drop(x_loc)

        # if self.STEM_RENORMALIZE:
        #     x_loc = F.normalize(x_loc, dim=-1)

        x_ctx = self.initMem.expand(x_loc.size())
        x_ctx_var_drop = generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(self.memoryDropout if self.training else 1.))
        return x_loc, x_ctx, x_ctx_var_drop
