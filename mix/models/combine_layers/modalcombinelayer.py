from ..builder import COMBINE_LAYERS
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


@COMBINE_LAYERS.register_module()
class ModalCombineLayer(nn.Module):

  def __init__(self, combine_type, img_feat_dim, txt_emb_dim, **kwargs):
    super().__init__()
    if combine_type == 'MFH':
      self.module = MFH(img_feat_dim, txt_emb_dim, **kwargs)
    elif combine_type == 'non_linear_element_multiply':
      self.module = NonLinearElementMultiply(img_feat_dim, txt_emb_dim,
                                             **kwargs)
    elif combine_type == 'two_layer_element_multiply':
      self.module = TwoLayerElementMultiply(img_feat_dim, txt_emb_dim, **kwargs)
    elif combine_type == 'top_down_attention_lstm':
      self.module = TopDownAttentionLSTM(img_feat_dim, txt_emb_dim, **kwargs)
    else:
      raise NotImplementedError('Not implemented combine type: %s' %
                                combine_type)

    self.out_dim = self.module.out_dim

  def forward(self, *args, **kwargs):
    return self.module(*args, **kwargs)


class MFH(nn.Module):

  def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
    super().__init__()
    self.mfb_expand_list = nn.ModuleList()
    self.mfb_sqz_list = nn.ModuleList()
    self.relu = nn.ReLU()

    hidden_sizes = kwargs['hidden_sizes']
    self.out_dim = int(sum(hidden_sizes) / kwargs['pool_size'])

    self.order = kwargs['order']
    self.pool_size = kwargs['pool_size']

    for i in range(self.order):
      mfb_exp_i = MfbExpand(
          img_feat_dim=image_feat_dim,
          txt_emb_dim=ques_emb_dim,
          hidden_dim=hidden_sizes[i],
          dropout=kwargs['dropout'],
      )
      self.mfb_expand_list.append(mfb_exp_i)
      self.mfb_sqz_list.append(self.mfb_squeeze)

  def forward(self, image_feat, question_embedding):
    feature_list = []
    prev_mfb_exp = 1

    for i in range(self.order):
      mfb_exp = self.mfb_expand_list[i]
      mfb_sqz = self.mfb_sqz_list[i]
      z_exp_i = mfb_exp(image_feat, question_embedding)
      if i > 0:
        z_exp_i = prev_mfb_exp * z_exp_i
      prev_mfb_exp = z_exp_i
      z = mfb_sqz(z_exp_i)
      feature_list.append(z)

    # append at last feature
    cat_dim = len(feature_list[0].size()) - 1
    feature = torch.cat(feature_list, dim=cat_dim)
    return feature

  def mfb_squeeze(self, joint_feature):
    # joint_feature dim: N x k x dim or N x dim

    orig_feature_size = len(joint_feature.size())

    if orig_feature_size == 2:
      joint_feature = torch.unsqueeze(joint_feature, dim=1)

    batch_size, num_loc, dim = joint_feature.size()

    if dim % self.pool_size != 0:
      exit('the dim %d is not multiply of \
             pool_size %d' % (dim, self.pool_size))

    joint_feature_reshape = joint_feature.view(batch_size, num_loc,
                                               int(dim / self.pool_size),
                                               self.pool_size)

    # N x 100 x 1000 x 1
    iatt_iq_sumpool = torch.sum(joint_feature_reshape, 3)

    iatt_iq_sqrt = torch.sqrt(self.relu(iatt_iq_sumpool)) - torch.sqrt(
        self.relu(-iatt_iq_sumpool))

    iatt_iq_sqrt = iatt_iq_sqrt.view(batch_size, -1)  # N x 100000
    iatt_iq_l2 = nn.functional.normalize(iatt_iq_sqrt)
    iatt_iq_l2 = iatt_iq_l2.view(batch_size, num_loc, int(dim / self.pool_size))

    if orig_feature_size == 2:
      iatt_iq_l2 = torch.squeeze(iatt_iq_l2, dim=1)

    return iatt_iq_l2


class NonLinearElementMultiply(nn.Module):

  def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
    super().__init__()
    self.fa_image = ReLUWithWeightNormFC(image_feat_dim, kwargs['hidden_dim'])
    self.fa_txt = ReLUWithWeightNormFC(ques_emb_dim, kwargs['hidden_dim'])

    context_dim = kwargs.get('context_dim', None)
    if context_dim is not None:
      self.fa_context = ReLUWithWeightNormFC(context_dim, kwargs['hidden_dim'])

    self.dropout = nn.Dropout(kwargs['dropout'])
    self.out_dim = kwargs['hidden_dim']

  def forward(self, image_feat, question_embedding, context_embedding=None):
    image_fa = self.fa_image(image_feat)
    question_fa = self.fa_txt(question_embedding)

    if len(image_feat.size()) == 3 and len(question_fa.size()) != 3:
      question_fa_expand = question_fa.unsqueeze(1)
    else:
      question_fa_expand = question_fa

    joint_feature = image_fa * question_fa_expand

    if context_embedding is not None:
      context_fa = self.fa_context(context_embedding)

      context_text_joint_feaure = context_fa * question_fa_expand
      joint_feature = torch.cat([joint_feature, context_text_joint_feaure],
                                dim=1)

    joint_feature = self.dropout(joint_feature)

    return joint_feature


class TopDownAttentionLSTM(nn.Module):

  def __init__(self, image_feat_dim, embed_dim, **kwargs):
    super().__init__()
    self.fa_image = weight_norm(
        nn.Linear(image_feat_dim, kwargs['attention_dim']))
    self.fa_hidden = weight_norm(
        nn.Linear(kwargs['hidden_dim'], kwargs['attention_dim']))
    self.top_down_lstm = nn.LSTMCell(
        embed_dim + image_feat_dim + kwargs['hidden_dim'],
        kwargs['hidden_dim'],
        bias=True,
    )
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(kwargs['dropout'])
    self.out_dim = kwargs['attention_dim']

  def forward(self, image_feat, embedding):
    image_feat_mean = image_feat.mean(1)

    # Get LSTM state
    state = registry.get(f'{image_feat.device}_lstm_state')
    h1, c1 = state['td_hidden']
    h2, c2 = state['lm_hidden']

    h1, c1 = self.top_down_lstm(
        torch.cat([h2, image_feat_mean, embedding], dim=1), (h1, c1))

    state['td_hidden'] = (h1, c1)

    image_fa = self.fa_image(image_feat)
    hidden_fa = self.fa_hidden(h1)

    joint_feature = self.relu(image_fa + hidden_fa.unsqueeze(1))
    joint_feature = self.dropout(joint_feature)

    return joint_feature


class TwoLayerElementMultiply(nn.Module):

  def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
    super().__init__()

    self.fa_image1 = ReLUWithWeightNormFC(image_feat_dim, kwargs['hidden_dim'])
    self.fa_image2 = ReLUWithWeightNormFC(kwargs['hidden_dim'],
                                          kwargs['hidden_dim'])
    self.fa_txt1 = ReLUWithWeightNormFC(ques_emb_dim, kwargs['hidden_dim'])
    self.fa_txt2 = ReLUWithWeightNormFC(kwargs['hidden_dim'],
                                        kwargs['hidden_dim'])

    self.dropout = nn.Dropout(kwargs['dropout'])

    self.out_dim = kwargs['hidden_dim']

  def forward(self, image_feat, question_embedding):
    image_fa = self.fa_image2(self.fa_image1(image_feat))
    question_fa = self.fa_txt2(self.fa_txt1(question_embedding))

    if len(image_feat.size()) == 3:
      num_location = image_feat.size(1)
      question_fa_expand = torch.unsqueeze(question_fa,
                                           1).expand(-1, num_location, -1)
    else:
      question_fa_expand = question_fa

    joint_feature = image_fa * question_fa_expand
    joint_feature = self.dropout(joint_feature)

    return joint_feature


class TransformLayer(nn.Module):

  def __init__(self, transform_type, in_dim, out_dim, hidden_dim=None):
    super().__init__()

    if transform_type == 'linear':
      self.module = LinearTransform(in_dim, out_dim)
    elif transform_type == 'conv':
      self.module = ConvTransform(in_dim, out_dim, hidden_dim)
    else:
      raise NotImplementedError('Unknown post combine transform type: %s' %
                                transform_type)
    self.out_dim = self.module.out_dim

  def forward(self, *args, **kwargs):
    return self.module(*args, **kwargs)


class LinearTransform(nn.Module):

  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.lc = weight_norm(
        nn.Linear(in_features=in_dim, out_features=out_dim), dim=None)
    self.out_dim = out_dim

  def forward(self, x):
    return self.lc(x)


class ConvTransform(nn.Module):

  def __init__(self, in_dim, out_dim, hidden_dim):
    super().__init__()
    self.conv1 = nn.Conv2d(
        in_channels=in_dim, out_channels=hidden_dim, kernel_size=1)
    self.conv2 = nn.Conv2d(
        in_channels=hidden_dim, out_channels=out_dim, kernel_size=1)
    self.out_dim = out_dim

  def forward(self, x):
    if len(x.size()) == 3:  # N x k xdim
      # N x dim x k x 1
      x_reshape = torch.unsqueeze(x.permute(0, 2, 1), 3)
    elif len(x.size()) == 2:  # N x dim
      # N x dim x 1 x 1
      x_reshape = torch.unsqueeze(torch.unsqueeze(x, 2), 3)

    iatt_conv1 = self.conv1(x_reshape)  # N x hidden_dim x * x 1
    iatt_relu = nn.functional.relu(iatt_conv1)
    iatt_conv2 = self.conv2(iatt_relu)  # N x out_dim x * x 1

    if len(x.size()) == 3:
      iatt_conv3 = torch.squeeze(iatt_conv2, 3).permute(0, 2, 1)
    elif len(x.size()) == 2:
      iatt_conv3 = torch.squeeze(torch.squeeze(iatt_conv2, 3), 2)

    return iatt_conv3


class BCNet(nn.Module):
  """Simple class for non-linear bilinear connect network."""

  def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=None, k=3):
    super().__init__()

    self.c = 32
    self.k = k
    self.v_dim = v_dim
    self.q_dim = q_dim
    self.h_dim = h_dim
    self.h_out = h_out
    if dropout is None:
      dropout = [0.2, 0.5]

    self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
    self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
    self.dropout = nn.Dropout(dropout[1])

    if k > 1:
      self.p_net = nn.AvgPool1d(self.k, stride=self.k)

    if h_out is None:
      pass

    elif h_out <= self.c:
      self.h_mat = nn.Parameter(
          torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
      self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
    else:
      self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

  def forward(self, v, q):
    if self.h_out is None:
      v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
      q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
      d_ = torch.matmul(v_, q_)
      logits = d_.transpose(1, 2).transpose(2, 3)
      return logits

    # broadcast Hadamard product, matrix-matrix production
    # fast computation but memory inefficient
    elif self.h_out <= self.c:
      v_ = self.dropout(self.v_net(v)).unsqueeze(1)
      q_ = self.q_net(q)
      h_ = v_ * self.h_mat
      logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2, 3))
      logits = logits + self.h_bias
      return logits

    # batch outer product, linear projection
    # memory efficient but slow computation
    else:
      v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
      q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
      d_ = torch.matmul(v_, q_)
      logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))
      return logits.transpose(2, 3).transpose(1, 2)

  def forward_with_weights(self, v, q, w):
    v_ = self.v_net(v).transpose(1, 2).unsqueeze(2)
    q_ = self.q_net(q).transpose(1, 2).unsqueeze(3)
    logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)
    logits = logits.squeeze(3).squeeze(2)

    if self.k > 1:
      logits = logits.unsqueeze(1)
      logits = self.p_net(logits).squeeze(1) * self.k

    return logits


class FCNet(nn.Module):
  """Simple class for non-linear fully connect network."""

  def __init__(self, dims, act='ReLU', dropout=0):
    super().__init__()

    layers = []
    for i in range(len(dims) - 2):
      in_dim = dims[i]
      out_dim = dims[i + 1]

      if dropout > 0:
        layers.append(nn.Dropout(dropout))

      layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))

      if act is not None:
        layers.append(getattr(nn, act)())

    if dropout > 0:
      layers.append(nn.Dropout(dropout))

    layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))

    if act is not None:
      layers.append(getattr(nn, act)())

    self.main = nn.Sequential(*layers)

  def forward(self, x):
    return self.main(x)


class BiAttention(nn.Module):

  def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=None):
    super().__init__()
    if dropout is None:
      dropout = [0.2, 0.5]
    self.glimpse = glimpse
    self.logits = weight_norm(
        BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
        name='h_mat',
        dim=None,
    )

  def forward(self, v, q, v_mask=True):
    p, logits = self.forward_all(v, q, v_mask)
    return p, logits

  def forward_all(self, v, q, v_mask=True):
    v_num = v.size(1)
    q_num = q.size(1)
    logits = self.logits(v, q)

    if v_mask:
      v_abs_sum = v.abs().sum(2)
      mask = (v_abs_sum == 0).unsqueeze(1).unsqueeze(3)
      mask = mask.expand(logits.size())
      logits.masked_fill_(mask, -float('inf'))

    expanded_logits = logits.view(-1, self.glimpse, v_num * q_num)
    p = nn.functional.softmax(expanded_logits, 2)

    return p.view(-1, self.glimpse, v_num, q_num), logits


class ReLUWithWeightNormFC(nn.Module):

  def __init__(self, in_dim, out_dim):
    super().__init__()

    layers = []
    layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
    layers.append(nn.ReLU())
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)
