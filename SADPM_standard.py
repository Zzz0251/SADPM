import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, SGD
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.version import __version__
import os
import numpy as np
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

ce_custom=torch.nn.CrossEntropyLoss()
def dice_custom(net_output, y_onehot, smooth):
    
    y_flag=torch.sum(torch.sum(torch.sum(y_onehot, axis=-1), axis=-1),axis=-1)

    if torch.sum(y_flag)>0:
        y_onehot=y_onehot[y_flag>0]
        net_output=net_output[y_flag>0]
        tp = torch.sum(net_output * y_onehot)
        fp = torch.sum(net_output * (1 - y_onehot))
        fn = torch.sum((1 - net_output) * y_onehot) 
        tn = torch.sum((1 - net_output) * (1 - y_onehot))
    
        dc = (2 * tp + smooth)/ (2 * tp + fp + fn + smooth + 1e-8)+ (tp+smooth) / (tp+fn+smooth + 1e-8)+ (tp+smooth) / (tp+fp+smooth + 1e-8)  # dice+sen+pre
    else:
        dc=torch.tensor([0], divice=net_output.device)
    return -dc


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)


        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

#  Unet2
class Unet2(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.ups2 = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim*2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
# for label

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups2.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block2 = block_klass(dim*2, dim, time_emb_dim = time_dim)
        self.final_conv2 = nn.Sequential(
        nn.Conv2d(dim, self.out_dim, 1),
        )
        #nn.Tanh(),
        self.final_act_y0=nn.Sequential(nn.Tanh(),)



    def forward(self, x, time1, time2, c1, c2, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
   

        
        
        
        x2 = self.init_conv(x)
        r2= x2.clone()
        x = self.init_conv(x)
        r = x.clone()
        
        
        
        t=self.time_mlp(time1)
        #t1 = self.time_mlp(time1)#   2,64
        #t2 = self.time_mlp(time2)
        

        
        #t=torch.cat((t1,t2), axis=1)
        # print('t: '+ str(t.shape))
        h = []
        g = []
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x2 = block1(x2, t)
            
            h.append(x)
            g.append(x2)
            
            x = block2(x, t)
            x2 = block2(x2, t)
            
            x = attn(x)
            x2 = attn(x2)
            
            h.append(x)
            g.append(x2)
            
            x = downsample(x)
            x2 = downsample(x2)
            
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        x2 = self.mid_block1(x2, t)
        x2 = self.mid_attn(x2)
        x2 = self.mid_block2(x2, t)
        
        
        
        
        y=x2

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        e = self.final_conv(x)
        
        
        for block1, block2, attn, upsample in self.ups2:
            y = torch.cat((y, g.pop()), dim = 1)
            y = block1(y, t)

            y = torch.cat((y, g.pop()), dim = 1)
            y = block2(y, t)
            y = attn(y)

            y = upsample(y)

        y = torch.cat((y, r2), dim = 1)

        y = self.final_res_block2(y, t)
        y = self.final_conv2(y)
        
        
        # return e, y, nn.Tanh(c2*e+c1*y)
        #  return e, y, c2*e+c1*y
        #print('c1: '+str(c1))
        #print('c1 shape: '+str(c1.shape))
        y0_est=y/c1-(c2/c1)*e
        #print('y0_est shape: '+str(y0_est.shape))
        # return e, self.final_act_y0(y0_est), y
        return e, y0_est, y

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        self_condition = False,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        loss_type_add_l2=False,
        loss_type_add_l1=False,
        add_posterior_noise=False,
        lamda2=0.5, 
        lamda1=0.95,
        posterior_D=3000,
        posterior_folder='',
        posterior_folder_result='',
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond


        self.posterior_folder=posterior_folder
        self.posterior_folder_result=posterior_folder_result

        self.loss_type_add_l2=loss_type_add_l2
        self.loss_type_add_l1=loss_type_add_l1

        self.add_posterior_noise=add_posterior_noise
        self.lamda2=lamda2
        self.lamda1=lamda1
        self.posterior_D=posterior_D
        
        
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32 

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output, _, _ = self.model(x, t, 0, extract(self.sqrt_alphas_cumprod, t, x.shape), extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape), x_self_cond)
        # model_output, _, _ = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
        
    def p_mean_variance_seg(self, x, y, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start
        pred_noise=preds.pred_noise
        y_start = self.predict_start_from_noise(y, t, pred_noise)
        
        if clip_denoised:
            x_start.clamp_(-1., 1.)
            y_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        model_meany, posterior_variancey, posterior_log_variancey = self.q_posterior(x_start = y_start, x_t = y, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start, model_meany, posterior_variancey, posterior_log_variancey, y_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
        
    @torch.no_grad()
    def p_sample_seg(self, x, y, t: int, x_self_cond = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        
        model_mean, _, model_log_variance, x_start, model_meany, _, model_log_variancey, y_start = self.p_mean_variance_seg(x = x, y=y,  t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        
        b,c,h,w=x.shape
        
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        
        
        
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        pred_imgy = model_meany + (0.5 * model_log_variancey).exp() * noise
        return pred_img, x_start, pred_imgy, y_start  
        

    @torch.no_grad() # using in function "sample"
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.betas.device


        if self.add_posterior_noise:
            b,c,h,w=x.shape



            img=self.posterior_noise(None, b, c, h, w, self.posterior_D, self.lamda1, self.lamda2, device)

        else:
            
            img = torch.randn(shape, device = device)
            
            
       
        
        imgs = [img]



        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
        
    @torch.no_grad()
    def p_sample_loop_seg(self, x_t, y_t, mask, t, return_all_timesteps = False):
        batch = x_t.shape[0]
        
        x_t=mask*x_t
        y_t=mask*y_t
        
        
        img=x_t
        imgs = [img]
        seg=y_t
        segs = [seg]
        
        x_start = None
        y_start = None
        tt=t[0].item()
        for t in tqdm(reversed(range(0, tt)), desc = 'segmenting loop time step', total = tt):
            self_cond = x_start if self.self_condition else None
            
            img, x_start, seg, y_start = self.p_sample_seg(img, seg, t, self_cond)
            
            
            img=mask*img
            x_start=mask*x_start
            seg=mask*seg
            y_start=mask*y_start
            
            
            imgs.append(img)
            segs.append(seg)

        ret = seg if not return_all_timesteps else torch.stack(segs, dim = 1)

        ret.clamp_(-1, 1)  
        ret = self.unnormalize(ret)
        return ret
        
    @torch.no_grad()
    def p_sample_loop_seg_union(self, x_t, y_t, mask, t, return_all_timesteps = False):
        batch = x_t.shape[0]
        
        x_t=mask*x_t
        y_t=mask*y_t
        
        
        img=x_t
        imgs = [img]
        seg=y_t
        segs = [seg]
        
        x_start = None
        y_start = None
        tt=t[0].item()
        
        alpha=1
        t_stop=10000
        for t in tqdm(reversed(range(0, tt)), desc = 'segmenting loop time step', total = tt):
            self_cond = x_start if self.self_condition else None
            
            img, x_start, seg, y_start = self.p_sample_seg(img, seg, t, self_cond)
            if t>t_stop:
                tp=torch.full((batch,), t, device = img.device, dtype = torch.long)
                _, y_0, y_t_1 = self.model(img, tp-1, 0, extract(self.sqrt_alphas_cumprod, tp, img.shape), extract(self.sqrt_one_minus_alphas_cumprod, tp, img.shape), None)
                

                y_start=y_start*alpha+(1-alpha)*(y_0)
            
            
            
            img=mask*img
            x_start=mask*x_start
            seg=mask*seg
            y_start=mask*y_start
            
            
            imgs.append(img)
            segs.append(seg)

        ret = seg if not return_all_timesteps else torch.stack(segs, dim = 1)

        ret.clamp_(-1, 1)  
        ret = self.unnormalize(ret)
        return ret  
            
        
        
        

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True)

            imgs.append(img)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()   
    def sample(self, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')



    def posterior_noise(self, x_start, bd, c, h, w, D, lamda, lamda2, device):
        
        folder=self.posterior_folder
        folder_result=self.posterior_folder_result
        accelerator = Accelerator(
            split_batches = True,
            mixed_precision = 'no'
        )
        files=os.listdir(folder)
        assert len(files)!=0
        ds = Dataset(folder, self.image_size, augment_horizontal_flip = True)
        dl = DataLoader(ds, batch_size = bd, shuffle = False, pin_memory = True, num_workers = 0) #8

        dl = accelerator.prepare(dl)
        dl = cycle(dl)        
        

        embs=[]
        d=c*h*w
        
        if d>5000:
            complete_w=False
        else:
            complete_w=True
        use_G=False
        force_execute=False
        
        
        if not 'W' in globals() and not 'cov_emb' in globals() and not 'b' in globals():
            global W
            global cov_emb
            global b
            
        
        
        
        
        if os.path.exists(folder_result+'b.pt') and (not force_execute) and not 'b' in globals():

            W=torch.load(folder_result+'W.pt')
            cov_emb=torch.load(folder_result+'cov_emb.pt')
            b=torch.load(folder_result+'b.pt')

            
        elif os.path.exists(folder_result+'b.pt') and (not force_execute) and 'b' in globals():
            pass

            
        else:
            alpha=math.sqrt(1)
            sigma_fft=math.sqrt(2)*10*torch.pi/(math.sqrt(D*d)*alpha)

            W=torch.randn(D,d, device=device)*sigma_fft                      # D x d
            b=(torch.rand(D,1, device=device)*0.5*torch.pi)+0.25*torch.pi

            for _ in range(len(files)):
                data = next(dl).to(device)
                
                data=(data[:,0])[:,None]
                

                n, c, h, w= data.shape
                d=c*h*w
                data=self.normalize(data)

                bm=b @ torch.ones([1,n], device=device)                     # D x n
                embs.append(torch.cos((W @ ((data).view(n,d)).T)+bm))   # dxn
                


            embs=torch.cat(embs, dim=1)
            cov_emb=torch.cov(embs) 
            

            
            torch.save(cov_emb, folder_result+'cov_emb.pt')
            torch.save(W, folder_result+'W.pt')
            torch.save(b, folder_result+'b.pt')
            

            
            
            del embs
 
        n=bd



        global e
        
        if not 'e' in globals():

            posterior_cov=lamda*cov_emb+(1-lamda)*torch.diag(torch.diag(cov_emb))+torch.eye(D, device=device)*1e-7
            e=torch.linalg.cholesky(posterior_cov)
        

        sample=e @ torch.randn(D,n, device=device)



        x_prime=0.1*sample+0.9*torch.std(sample)*torch.randn_like(sample, device=device)





        x_prime.clamp_(-1, 1)
        bm=b @ torch.ones([1,n], device=device) # D x n
        x_prime=torch.arccos(x_prime)-bm   

        
        if complete_w:
            WW=W.T @ W
            inv_WW=torch.linalg.inv(WW+torch.eye(W.shape[1])*1e-12)   # this is feasible
        
        if use_G:
            G=torch.linalg.pinv(W)
        
        
        if d<D:
            
            I_re=inv_WW @ WW
            x_re=inv_WW @ (W.T @ x_prime)

        else:
            
            if complete_w:
                x_re= (inv_WW @ (W.T @ x_prime))/(d/(d+D))
            else:
                if use_G:
                    x_re=G @ x_prime
                else:
                    x_re= W.T @ x_prime     # W.T:d x D    ,X_prime # D x n
            


        x_re_mean=torch.ones([d,1], device=device) @ torch.mean(x_re, axis=0)[None]
        std_val=torch.ones([d,1], device=device) @ torch.std(x_re, axis=0)[None]
        

        x_re=(x_re-x_re_mean)/std_val
        


        x_re=(self.lamda2)*x_re+(1-self.lamda2)*torch.randn_like(x_re, device=device)

        x_re=(x_re.T).view(n, c, h, w)
        
        

        
        return x_re


    def p_losses(self, x_start, x_start_seg, t,t2, noise = None):
        b, c, h, w = x_start.shape

        mask=(x_start>-1).to(x_start.dtype)

        device=x_start.device
        if self.add_posterior_noise:

            noise=self.posterior_noise(x_start, b, c, h, w, self.posterior_D, self.lamda1, self.lamda2, device)

        else:
            noise=torch.randn_like(x_start)
        
        
        
        
        
        
        
        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        y = self.q_sample(x_start = x_start_seg, t = t, noise = noise)
        
        x=x*mask
        y=y*mask
        x_start_seg=x_start_seg*mask
        noise=noise*mask
        
        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step



        model_out, model_seg, model_seg_noise = self.model(x, t, t2, extract(self.sqrt_alphas_cumprod, t, x.shape), extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape), x_self_cond)

        #model_out=model_out*mask
        #model_seg=model_seg*mask
        #model_seg_noise=model_seg_noise*mask
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')


        if self.loss_type_add_l2:
            loss = self.loss_fn(model_out, target, reduction = 'none')+0.1*F.mse_loss(model_out, target, reduction = 'none')
        elif self.loss_type_add_l1:
            loss = self.loss_fn(model_out, target, reduction = 'none')+0.1*F.l1_loss(model_out, target, reduction = 'none')  
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none')

        
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        
        
        loss2= self.loss_fn(model_seg_noise, y, reduction = 'none')
        loss2 = reduce(loss2, 'b ... -> b (...)', 'mean')   # (self.p2_loss_weight, t, loss.shape) wait to test.
        



        return loss.mean()+loss2.mean()+F.mse_loss(model_seg, x_start_seg) + dice_custom(torch.sigmoid(model_seg), (x_start_seg+1)/2, 1e-8)# +loss_orth   # torch.sigmoid(model_seg+1) is wrong?
        #return loss2.mean()

        
        #return loss.mean()+dice_custom((model_seg+1)/2, (x_start_seg+1)/2, 1e-8)+ce_custom((model_seg+1)/2, (x_start_seg+1)/2)  +F.mse_loss(model_seg, x_start_seg)
        # return loss.mean()  +dice_custom((model_seg.clamp_(-1, 1)+1)/2, (x_start_seg+1)/2, 1e-8)
        
        
    def forward(self, data, t2, *args, **kwargs):

        
        img=(data[:,0])[:,None]
        seg=(data[:,1])[:,None]
        
        
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
    
        
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        
        img = self.normalize(img)
        seg = self.normalize(seg)
        return self.p_losses(img, seg, t, t2, *args, **kwargs)

# dataset classes

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff', 'npy'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([

            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)

        tf=self.transform(img)

        
        # data naming
        path_str=str(path)
        zt=int(path_str[-8:-4]) # [-9:-4]  ValueError: invalid literal for int() with base 10: '_1749'
        #print(zt)
        tf[2]=zt
        if torch.max(tf[1])>0.8 and torch.max(tf[1])<1:
            tf[tf>0.8]=1
            print('happen!')

        return tf

class Dataset2(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff', 'npy'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__(folder, image_size)
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]


        self.transform = T.Compose([

            T.CenterCrop(image_size),
            #T.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)

        tf=self.transform(img)

        
        path_str=str(path)
        zt=int(path_str[-8:-4]) #[-9:-4]
        #print(zt)
        tf[2]=zt
        if torch.max(tf[1])>0.8 and torch.max(tf[1])<1:
            tf[tf>0.8]=1
            print('happen!')

        return tf



# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        optimizer_type='Adam',
        results_folder_absolute='',
        min_training_lr=1e-6,
        training_schedual='linear_decrease',
        warmup_first=1000,
        warmup_second=2000,
        plot_folder='',
        if_posterior_predict=False,
        lamda2=0, 
        lamda1=0.5,
        posterior_D=3000,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.min_training_lr=min_training_lr
        self.training_schedual=training_schedual
        self.train_lr=train_lr
        self.warmup_first=warmup_first
        self.warmup_second=warmup_second
        self.if_posterior_predict=if_posterior_predict

        self.lamda2=lamda2
        self.lamda1=lamda1
        self.posterior_D=posterior_D



        if not os.path.exists(results_folder_absolute):
            os.mkdir(results_folder_absolute)





        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.optimizer_type=optimizer_type
        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 0)  # num_workers = cpu_count()

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.ds2 = Dataset2(plot_folder, self.image_size, augment_horizontal_flip = False)
        dl2 = DataLoader(self.ds2, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 0)

        dl2 = self.accelerator.prepare(dl2)
        self.dl2 = cycle(dl2)
        
        

        # optimizer
        if self.optimizer_type=='Adam':
            self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas, eps=1e-8)
        elif self.optimizer_type=='SGD':
            self.opt = SGD(diffusion_model.parameters(), lr = train_lr, momentum=0.9)
        else:
            raise ValueError(f'unknown optimizer {self.optimizer_type}')
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.results_folder_absolute=results_folder_absolute
        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }


        torch.save(data, self.results_folder_absolute+'/model-latest.pt')
        
        
        if milestone%100==0 and milestone!=0:

            torch.save(data, self.results_folder_absolute+f'/model-{milestone}.pt')

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device


        data = torch.load(self.results_folder_absolute+'/model-latest.pt', map_location=device)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])




    def predict(self, folder_testing, folder_testing_out, cal_dice, if_eliminate_discrete):

        device=self.accelerator.device
        self.ds3 = Dataset2(folder_testing, self.image_size, augment_horizontal_flip = False)
        dl3 = DataLoader(self.ds3, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0) #1
        dl3 = self.accelerator.prepare(dl3)
        self.dl3 = cycle(dl3)
        
        n_files=os.listdir(folder_testing)
    
        if not os.path.exists(folder_testing_out):  
            os.mkdir(folder_testing_out)
    
    
        
        if cal_dice:
            log_path=folder_testing_out+'/dice_report.txt'
            if not os.path.exists(log_path):  
                f = open(log_path,'w')
                f.close()
            
            
        def L_to_RGB(x,y,z):
            y=torch.cat((x,y,z), axis=1)
            return y
            
        if if_eliminate_discrete:
            eliminate_discrete=nn.Conv2d(1, 1, 7, padding = 3, bias=False)
            # if ker=3, pad=1; if ker=5, pad=2, if ker=7, pad=3
            
            #eliminate_discrete.state_dict()
            sd = eliminate_discrete.state_dict()
            sd['weight']=torch.ones_like(eliminate_discrete.weight).to(eliminate_discrete.weight.dtype)
            eliminate_discrete.load_state_dict(sd)
            # print(eliminate_discrete.weight)
            eliminate_discrete.to(device)
                        
                        
            
            
        dcs=[]
        cn=0
        
        
        for nfile in n_files:
            print(nfile)
            data_test = (next(self.dl3).to(device)).to(torch.float32)
            # b, c, h, w
            cn=cn+1
            
            if cn>-1:   # cn>920
                b=data_test.shape[0]    # here, b == 1
    
                img_test=(data_test[:,0])[:,None]
                seg_test=(data_test[:,1])[:,None]
    
                img_test=self.ema.ema_model.normalize(img_test)
    
    
    
                if not os.path.exists(os.path.join(folder_testing_out, 'seg_loop')):
                    os.mkdir(os.path.join(folder_testing_out, 'seg_loop'))
        
                    
                if not os.path.exists(os.path.join(folder_testing_out, 'salient_coef')):
                    os.mkdir(os.path.join(folder_testing_out, 'salient_coef'))

                if not os.path.exists(os.path.join(folder_testing_out, 'direct_y0')):
                    os.mkdir(os.path.join(folder_testing_out, 'direct_y0')) 

                if not os.path.exists(os.path.join(folder_testing_out, 'gt')):
                    os.mkdir(os.path.join(folder_testing_out, 'gt'))
                    
    
                image_gt=L_to_RGB(seg_test,seg_test,seg_test)
                utils.save_image(image_gt, os.path.join(folder_testing_out, 'gt', nfile), nrow = 1)    
    
                
                t=self.ema.ema_model.num_timesteps//20
                t=torch.full((b,), t, device = data_test.device, dtype = torch.long)
                
                b, c, h, w= img_test.shape
                seg_loops=[]
                for loop in range(0,60):
                    if self.if_posterior_predict:
        
                        noise=self.ema.ema_model.posterior_noise(img_test, b, c, h, w, self.posterior_D, self.lamda1, self.lamda2, device)
                    else:
                        noise=torch.randn_like(img_test)
                    
                    
    
                    x_t=self.ema.ema_model.q_sample(x_start=img_test, t=t, noise=noise)
                    mask=(img_test!=-1).to(img_test.dtype)
                    x_t=x_t*mask
                    # y_t=q_sample(x_start=seg_test, t=t, noise=noise)   
                    _, y_0, y_t = self.ema.ema_model.model(x_t, t, 0, extract(self.ema.ema_model.sqrt_alphas_cumprod, t, img_test.shape), extract(self.ema.ema_model.sqrt_one_minus_alphas_cumprod, t, img_test.shape), None)   # 0 is useless for now
                    
                    
                    seg_loops.append(self.ema.ema_model.p_sample_loop_seg(x_t, y_t, mask, t, return_all_timesteps = False))
                seg_loop=torch.mean(torch.cat(seg_loops, dim=0), dim=0)[None]
  
                
                

                # seg_loop is num_timesteps//2, seg_loop2 is num_timesteps//4, seg_loop3 is num_timesteps//8

                
                if if_eliminate_discrete:

                    sl1_input=((seg_loop[0,0])>0.5)[None, None].to(torch.float32)
                    sl1_cnn=eliminate_discrete(sl1_input)*sl1_input
                    sl1=(sl1_cnn>1).to(torch.float32)
                    
                    
                else:
                    sl1=seg_loop[0,0][None, None]

                # print(sl1.shape)
                image1=L_to_RGB((((sl1))>0.5).to(torch.float32), (((sl1))>0.5).to(torch.float32), (((sl1))>0.5).to(torch.float32))
                utils.save_image(image1, os.path.join(folder_testing_out, 'seg_loop', nfile), nrow = 1)


                t_start=self.ema.ema_model.num_timesteps//8
                coef=np.array(range(0,t_start))
                coef=(np.max(coef)-coef)/2
                coef=coef/np.sum(coef)
                

                y_00=torch.zeros_like(img_test, device = data_test.device)

                
                
                y_00s=[]
                y2s=[]
                for loop in range(0, 30):
                
                    for t in range(0,t_start, 1):
                        t2=t
                        t=torch.full((b,), t, device = data_test.device, dtype = torch.long)
                        
                        
                        if self.if_posterior_predict:
        
                            noise=self.ema.ema_model.posterior_noise(img_test, b, c, h, w, self.posterior_D, self.lamda1, self.lamda2, device)
                        else:
                            noise=torch.randn_like(img_test)
                        
                        
                        
                        x_t=self.ema.ema_model.q_sample(x_start=img_test, t=t, noise=noise)
                        mask=(img_test>-1).to(img_test.dtype) 
                        x_t=x_t*mask

                        noise_estimate, _, y_t = self.ema.ema_model.model(x_t, t, 0, extract(self.ema.ema_model.sqrt_alphas_cumprod, t, img_test.shape), extract(self.ema.ema_model.sqrt_one_minus_alphas_cumprod, t, img_test.shape), None)   # 0 is useless for now
                        
                        y_0=self.ema.ema_model.predict_start_from_noise(y_t, t, noise_estimate)
                        y_0.clamp_(-1, 1)
                        y_0 = self.ema.ema_model.unnormalize(y_0)
                        
                        if t2==0:
                            y2=y_0*mask

                            
                        if t2<t_start:
                            y_00=mask*y_0*coef[t2]+y_00


                    y_00s.append(y_00)
                    y2s.append(y2)
                y_00=torch.mean(torch.cat(y_00s, dim=0), dim=0)[None]
                y2=torch.mean(torch.cat(y2s, dim=0), dim=0)[None]
                
                
                
                
                if if_eliminate_discrete:

                    sl1_input=((y_00[0,0])>0.5)[None, None].to(torch.float32)
                    sl1_cnn=eliminate_discrete(sl1_input)*sl1_input
                    sl1=(sl1_cnn>1).to(torch.float32)
                    
                    
                    sl3_input=((y2[0,0])>0.5)[None, None].to(torch.float32)
                    sl3_cnn=eliminate_discrete(sl3_input)*sl3_input
                    sl3=(sl3_cnn>1).to(torch.float32)
                else:
                    sl1=y_00[0,0][None, None]

                    sl3=y2[0,0][None, None]
                
                image4=L_to_RGB((((sl1))>0.5).to(torch.float32), (((sl1))>0.5).to(torch.float32), (((sl1))>0.5).to(torch.float32))
                utils.save_image(image4, os.path.join(folder_testing_out, 'salient_coef', nfile), nrow = 1)

                image6=L_to_RGB((((sl3))>0.5).to(torch.float32), (((sl3))>0.5).to(torch.float32), (((sl3))>0.5).to(torch.float32))
                utils.save_image(image6, os.path.join(folder_testing_out, 'direct_y0', nfile), nrow = 1)
                

                
                
                
                def dice_testing(net_output, y_onehot):
                    tp = torch.sum(net_output * y_onehot)
                    fp = torch.sum(net_output * (1 - y_onehot))
                    fn = torch.sum((1 - net_output) * y_onehot)
                    tn = torch.sum((1 - net_output) * (1 - y_onehot))
                
                    dc = (2 * tp)/ (2 * tp + fp + fn + 1e-8)
                    return dc
                
                if cal_dice:
                    gt=torch.squeeze(image_gt)
                    predicted_img1=image1[:,0]

                    predicted_img4=image4[:,0]

                    predicted_img6=image6[:,0]


                    
                    if torch.sum(gt):
                        dc1=dice_testing(predicted_img1, gt)

                        dc4=dice_testing(predicted_img4, gt)

                        dc6=dice_testing(predicted_img6, gt)
                    

                                             
                        dcs.append([dc1.item(), dc4.item(), dc6.item()])
                        print_text_result=f'{cn}/{len(n_files)}, {nfile}: {dc1:.3f}, {dc4:.3f}, {dc6:.3f}'
                        f = open(log_path,'a')
                        f.write('\n'+print_text_result)
                        f.close()
                    else:
                        print_text_result=f'{cn}/{len(n_files)}, {nfile}: gt has no foreground.'
                        f = open(log_path,'a')
                        f.write('\n'+print_text_result)
                        f.close()
                    
        
        if cal_dice:
            dcs_np=np.array(dcs)
            dcs_np_mean=np.mean(dcs_np, axis=0)
            np.save(folder_testing_out+'/dices.npy', dcs_np)
            print_text_result=f'---------------------------\n average dice: {dcs_np_mean[0]:.3f}, {dcs_np_mean[3]:.3f}, {dcs_np_mean[5]:.3f}'
            f = open(log_path,'a')
            f.write('\n'+print_text_result)
            f.close()


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        log_path=self.results_folder_absolute+'/progress.txt'
        if not os.path.exists(log_path):  
            f = open(log_path,'w')
            f.close()

        avg_loss=torch.ones(self.save_and_sample_every)

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                # torch.cuda.empty_cache()
                total_loss = 0.

                if self.training_schedual=='linear_decrease':
                    cur_lr=self.train_lr*((1-self.step/self.train_num_steps)**0.9)+self.min_training_lr*((self.step/self.train_num_steps)**0.9)
                    self.opt.param_groups[0]['lr']=cur_lr
                    
                elif self.training_schedual=='cosine_annealing':
                    if self.step<self.warmup_first:
                        cur_lr=self.min_training_lr+0.5*(self.train_lr-self.min_training_lr)*(1+np.cos((self.step/self.warmup_first)*np.pi))
                    elif self.step>=self.warmup_first and self.step<self.warmup_second:
                        cur_lr=self.min_training_lr+0.5*(self.train_lr-self.min_training_lr)*(1+np.cos(((self.step-self.warmup_first)/(self.warmup_second-self.warmup_first))*np.pi))
                    elif self.step>=self.warmup_second:
                        cur_lr=self.min_training_lr+0.5*(self.train_lr-self.min_training_lr)*(1+np.cos(((self.step-self.warmup_second)/(self.train_num_steps-self.warmup_second))*np.pi))
                    else:
                        raise ValueError(f'unknown error in consine anealing.')
                    self.opt.param_groups[0]['lr']=cur_lr
                elif self.training_schedual=='consistant':
                    
                    cur_lr=self.opt.param_groups[0]['lr']
                else:
                    raise ValueError(f'unknown training schedual.')
                    
                
                
                for _ in range(self.gradient_accumulate_every):
                    data = (next(self.dl).to(device)).to(torch.float32)
                    t2=torch.mean(data[:,2],axis=-1)
                    t2=(torch.mean(t2,axis=-1)).to(torch.int64)
                    
                    #print(t2.shape)
                    #print(t2)
                    
                    with self.accelerator.autocast():
                        loss = self.model(data, t2)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        if torch.mean(avg_loss)==1:
                            avg_loss=avg_loss*total_loss
                        avg_loss[self.step % self.save_and_sample_every]=total_loss

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                print_text_result=f'loss: {total_loss:.4f}, lr:{cur_lr:.12f}, avg_loss over {self.save_and_sample_every}: {torch.mean(avg_loss):.6f}'
                pbar.set_description(print_text_result)



                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                def L_to_RGB(x,y,z):
                    y=torch.cat((x,y,z), axis=1)
                    return y



                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        f = open(log_path,'a')
                        f.write('\n'+print_text_result)
                        f.close()
                        if self.ema.ema_model.num_timesteps==self.ema.ema_model.sampling_timesteps:
                            with torch.no_grad():
                                milestone = self.step // self.save_and_sample_every
                            
                            data_test = (next(self.dl2).to(device)).to(torch.float32)
                            # b, c, h, w
                            # b=data_test.shape[0]
                            
                            img_test=(data_test[:,0])[:,None]
                            seg_test=(data_test[:,1])[:,None]
                            b, c, h, w= img_test.shape
                            img_test=self.ema.ema_model.normalize(img_test)
                            seg_test=self.ema.ema_model.normalize(seg_test)
                            
                            
                            
                            
                            t=self.ema.ema_model.num_timesteps//4
                            t=torch.full((b,), t, device = data_test.device, dtype = torch.long)
                            
                            if self.if_posterior_predict:

                                noise=self.ema.ema_model.posterior_noise(img_test, b, c, h, w, self.posterior_D, self.lamda1, self.lamda2, device)
                            else:
                                noise=torch.randn_like(img_test)
                            
                            
                            
                            x_t=self.ema.ema_model.q_sample(x_start=img_test, t=t, noise=noise)
                            mask=(img_test!=-1).to(img_test.dtype)
                            x_t=x_t*mask
                            # y_t=q_sample(x_start=seg_test, t=t, noise=noise)   
                            _, y_0, y_t = self.ema.ema_model.model(x_t, t, 0, extract(self.ema.ema_model.sqrt_alphas_cumprod, t, img_test.shape), extract(self.ema.ema_model.sqrt_one_minus_alphas_cumprod, t, img_test.shape), None)   # 0 is useless for now
                            
                            seg_loop=self.ema.ema_model.p_sample_loop_seg(x_t, y_t, mask, t, return_all_timesteps = False)
                            
                            
                            t=self.ema.ema_model.num_timesteps//2
                            t=torch.full((b,), t-1, device = data_test.device, dtype = torch.long)
                            
                            x_t=self.ema.ema_model.q_sample(x_start=img_test, t=t, noise=noise)

                            x_t=x_t*mask
                            # y_t=q_sample(x_start=seg_test, t=t, noise=noise)   
                            _, y_0, y_t = self.ema.ema_model.model(x_t, t, 0, extract(self.ema.ema_model.sqrt_alphas_cumprod, t, img_test.shape), extract(self.ema.ema_model.sqrt_one_minus_alphas_cumprod, t, img_test.shape), None)   # 0 is useless for now
                            
                            seg_loop2=self.ema.ema_model.p_sample_loop_seg(x_t, y_t, mask, t, return_all_timesteps = False)
                            
                            
                            t=self.ema.ema_model.num_timesteps//8
                            t=torch.full((b,), t-1, device = data_test.device, dtype = torch.long)
                            
                            x_t=self.ema.ema_model.q_sample(x_start=img_test, t=t, noise=noise)

                            x_t=x_t*mask
                            # y_t=q_sample(x_start=seg_test, t=t, noise=noise)   
                            _, y_0, y_t = self.ema.ema_model.model(x_t, t, 0, extract(self.ema.ema_model.sqrt_alphas_cumprod, t, img_test.shape), extract(self.ema.ema_model.sqrt_one_minus_alphas_cumprod, t, img_test.shape), None)   # 0 is useless for now
                            
                            seg_loop3=self.ema.ema_model.p_sample_loop_seg(x_t, y_t, mask, t, return_all_timesteps = False)
                            
                            
                            
                            
                            all_images_list=[]
                            for r in range(0,2):
                                all_images_list.append(L_to_RGB((data_test[r,0])[None, None], (data_test[r,0])[None, None], (data_test[r,0])[None, None]))
                                all_images_list.append(L_to_RGB((data_test[r,0])[None, None]*(1-(data_test[r,1])[None, None]), (data_test[r,1])[None, None], torch.zeros_like((data_test[r,1])[None, None])))
                                all_images_list.append(L_to_RGB(torch.zeros_like((y_t[r,0])[None, None]), (data_test[r,1])[None, None], (((seg_loop3[r,0])[None, None])>0.5).to(torch.float32)))
                                all_images_list.append(L_to_RGB(torch.zeros_like((y_t[r,0])[None, None]), (data_test[r,1])[None, None], (((seg_loop[r,0])[None, None])>0.5).to(torch.float32)))
                                all_images_list.append(L_to_RGB(torch.zeros_like((y_0[r,0])[None, None]), (data_test[r,1])[None, None], (((seg_loop2[r,0])[None, None])>0.5).to(torch.float32)))
                            all_images = torch.cat(all_images_list, dim = 0)   
                            # print(all_images.shape)
                            utils.save_image(all_images, os.path.join(self.results_folder_absolute, f'sample-loop-{milestone}.png'), nrow = 2)
                            

                            
                            coef=np.array(range(0,self.ema.ema_model.num_timesteps))
                            coef=(np.max(coef)-coef)/2
                            coef=coef/np.sum(coef)
                            
                            cc=0
                            y_00=torch.zeros_like(img_test, device = data_test.device)
                            y_01=torch.zeros_like(img_test, device = data_test.device)
                            for t in range(0,self.ema.ema_model.num_timesteps, 1):
                                t2=t
                                t=torch.full((b,), t, device = data_test.device, dtype = torch.long)
                                
                                if self.if_posterior_predict:

                                    noise=self.ema.ema_model.posterior_noise(img_test, b, c, h, w, self.posterior_D, self.lamda1, self.lamda2, device)
                                else:   
                                    noise=torch.randn_like(img_test)
                                
                                
                                x_t=self.ema.ema_model.q_sample(x_start=img_test, t=t, noise=noise)
                                mask=(img_test>-1).to(img_test.dtype) 
                                x_t=x_t*mask
                                # y_t=q_sample(x_start=seg_test, t=t, noise=noise)   
                                noise_estimate, _, y_t = self.ema.ema_model.model(x_t, t, 0, extract(self.ema.ema_model.sqrt_alphas_cumprod, t, img_test.shape), extract(self.ema.ema_model.sqrt_one_minus_alphas_cumprod, t, img_test.shape), None)   # 0 is useless for now
                                
                                y_0=self.ema.ema_model.predict_start_from_noise(y_t, t, noise_estimate)
                                y_0.clamp_(-1, 1)
                                y_0 = self.ema.ema_model.unnormalize(y_0)
                                
                                if t2==0:
                                    y2=y_0*mask
                                y_00=mask*y_0*coef[t2]+y_00
                                y_01=y_01+y_0*mask
                                cc=cc+1
                            y_01=y_01/cc
                            
                            
                            all_images_list=[]
                            for r in range(0,2):
                                all_images_list.append(L_to_RGB((data_test[r,0])[None, None], (data_test[r,0])[None, None], (data_test[r,0])[None, None]))
                                all_images_list.append(L_to_RGB((data_test[r,0])[None, None]*(1-(data_test[r,1])[None, None]), (data_test[r,1])[None, None], torch.zeros_like((data_test[r,1])[None, None])))
                                all_images_list.append(L_to_RGB(torch.zeros_like((y_t[r,0])[None, None]), (data_test[r,1])[None, None], (((y_00[r,0])[None, None])>0.5).to(torch.float32)))
                                all_images_list.append(L_to_RGB(torch.zeros_like((y_0[r,0])[None, None]), (data_test[r,1])[None, None], (((y_01[r,0])[None, None])>0.5).to(torch.float32)))
                                all_images_list.append(L_to_RGB(torch.zeros_like((y_0[r,0])[None, None]), (data_test[r,1])[None, None], (((y2[r,0])[None, None])>0.5).to(torch.float32)))
                            all_images = torch.cat(all_images_list, dim = 0)   
                            # print(all_images.shape)
                            utils.save_image(all_images, os.path.join(self.results_folder_absolute, f'sample-noloop-avg-{milestone}.png'), nrow = 2)   
                            

                        
                        self.save(milestone)
                
                pbar.update(1)

        accelerator.print('training complete')
