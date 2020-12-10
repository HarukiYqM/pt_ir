
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

import common.images
import common.metrics
import models


def update_argparser(parser):
    models.update_argparser(parser)
    args, _ = parser.parse_known_args()
    parser.add_argument(
      '--n_resblocks',
      help='Number of residual blocks in networks.',
      default=32,
      type=int)
    parser.add_argument(
      '--n_feats',
      help='Channel width.',
      default=256,
      type=int)
    parser.add_argument(
      '--res_scale',
      help='rescale residual connections',
      default=0.1,
      type=float)
    parser.add_argument(
      '--rgb_range',
      help='rgb value range',
      default=1,
      type=int)
    parser.add_argument(
      '--n_colors',
      help='rgb channel',
      default=3,
      type=int)
    if args.dataset.startswith('div2k'):
        parser.set_defaults(
        train_epochs=32,
        learning_rate_milestones=(8,16,24),
        learning_rate_decay=0.5,
        save_checkpoints_epochs=1,
        lr_patch_size=48,
        train_temporal_size=1,
        eval_temporal_size=1,
    )
    else:
        raise NotImplementedError('Needs to tune hyper parameters for new dataset.')


def get_model_spec(params):
    model = MODEL(params)
    print('# of parameters: ', sum([p.numel() for p in model.parameters()]))
    optimizer = optim.Adam(model.parameters(), params.learning_rate)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                params.learning_rate_milestones,
                                                params.learning_rate_decay)
    loss_fn = torch.nn.L1Loss()
    metrics = {
      'loss':
          loss_fn,
      'PSNR':
          functools.partial(
              common.metrics.psnr,
              shave=0 if params.scale == 1 else params.scale + 6),
      'PSNR_Y':
          functools.partial(
              common.metrics.psnr_y,
              shave=0 if params.scale == 1 else params.scale),
    }
    return model, loss_fn, optimizer, lr_scheduler, metrics

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return PANET(args, dilated.dilated_conv)
    else:
        return PANET(args)


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=(kernel_size//2),stride=stride, bias=bias)

class MODEL(nn.Module):
    def __init__(self, args, conv= default_conv):
        super(MODEL, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.msa = PyramidAttention(channel=n_feats, reduction=8,res_scale=args.res_scale);         
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock//2)
        ]
        m_body.append(self.msa)
        for _ in range(n_resblock//2):
            m_body.append(ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class PyramidAttention(nn.Module):
    def __init__(self, level=5, res_scale=1, channel=64, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True, conv=default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1-i/10 for i in range(level)]
        self.reduction = reduction
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = BasicBlock(conv,channel,channel//reduction, 3, bn=False, act=None)
        self.conv_match = BasicBlock(conv,channel, channel//reduction, 3, bn=False, act=None)
        self.conv_assembly = BasicBlock(conv,channel, channel,1,bn=False, act=None)

    def forward(self, input):
        res = input
        N,C,H,W = input.shape
        #theta
        match_base = self.conv_match_L_base(input)

        # patch size for matching 
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        #build feature pyramid
        for i in range(len(self.scale)):    
            ref = input
            if self.scale[i]!=1:
                ref  = F.interpolate(input, scale_factor=self.scale[i], mode='bicubic')
            #feature transformation function f

            base = self.conv_assembly(ref)
            base = torch.reshape(base,[N,C,-1])
            raw_w.append(base)

            #feature transformation function g
            ref_i = self.conv_match(ref)
            ref_i = torch.reshape(ref_i,[N,C//self.reduction,-1])
            w.append(ref_i)
        
        match_pyramid = torch.cat(w,dim=-1)
        match_raw = torch.cat(raw_w,dim=-1).permute(0,2,1)
        match_base = torch.reshape(match_base,[N,C//self.reduction,-1]).permute(0,2,1)
        score = F.softmax(torch.matmul(match_base,match_pyramid),dim=-1)
        y = torch.matmul(score,match_raw)
        y = torch.reshape(y,[N,C,H,W])
        y = y*self.res_scale+res  # back to the mini-batch
        return y


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=(kernel_size//2),stride=stride, bias=bias)

class PANET(nn.Module):
    def __init__(self, args, conv= default_conv):
        super(PANET, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.msa = PyramidAttention(channel=64, reduction=4,res_scale=args.res_scale);         
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock//2)
        ]
        m_body.append(self.msa)
        for _ in range(n_resblock//2):
            m_body.append(ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class PyramidAttention(nn.Module):
    def __init__(self, level=5, res_scale=1, channel=64, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True, conv=default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1-i/10 for i in range(level)]
        self.reduction = reduction
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = BasicBlock(conv,channel,channel//reduction, 3, bn=False, act=None)
        self.conv_match = BasicBlock(conv,channel, channel//reduction, 3, bn=False, act=None)
        self.conv_assembly = BasicBlock(conv,channel, channel,1,bn=False, act=None)

    def forward(self, input):
        res = input
        N,C,H,W = input.shape
        #theta
        match_base = self.conv_match_L_base(input)

        # patch size for matching 
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        #build feature pyramid
        for i in range(len(self.scale)):    
            ref = input
            if self.scale[i]!=1:
                ref  = F.interpolate(input, scale_factor=self.scale[i], mode='bicubic')
            #feature transformation function f

            base = self.conv_assembly(ref)
            base = torch.reshape(base,[N,C,-1])
            raw_w.append(base)

            #feature transformation function g
            ref_i = self.conv_match(ref)
            ref_i = torch.reshape(ref_i,[N,C//self.reduction,-1])
            w.append(ref_i)
        
        match_pyramid = torch.cat(w,dim=-1)
        match_raw = torch.cat(raw_w,dim=-1).permute(0,2,1)
        match_base = torch.reshape(match_base,[N,C//self.reduction,-1]).permute(0,2,1)
        score = F.softmax(torch.matmul(match_base,match_pyramid),dim=-1)
        y = torch.matmul(score,match_raw)
        y = torch.reshape(y,[N,C,H,W])
        y = y*self.res_scale+res  # back to the mini-batch
        return y


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

