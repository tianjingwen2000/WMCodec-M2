# *_*coding:utf-8 *_*
"""
adapted from: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
"""
import copy

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce
from position_embedding import PositionalEmbedding

import pdb

# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


# Note: GEGLU() is different from that (i.e., GELU()) in mbt.py
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), 
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim) # if context_dim exist, q-v from context_dim, otherwise from query_dim-self 

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h) # (B*h, 1, T2)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

    
# half MPU (mutual promotion unit)
class EMT_Fusion(nn.Module):
    def __init__(self, latent_dim, input_dim, depth, heads, dim_head, ff_expansion=4, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.mpu_list = nn.ModuleList([])
        for _ in range(depth):
            self.mpu_list.append(nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, context_dim=input_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                        context_dim=input_dim),
                PreNorm(latent_dim, FeedForward(latent_dim, mult=ff_expansion, dropout=ff_dropout))
            ]))

    def forward(self, x, context, mask=None, context_mask=None):
        for cross_attn, ff in self.mpu_list:
            x = cross_attn(x, context=context, mask=context_mask) + x
            x = ff(x) + x
        return x


if __name__ == "__main__":
    latent_dim = 512
    input_dim = 256
    depth = 2
    heads = 4
    model = EMT_Fusion(latent_dim = latent_dim, input_dim = input_dim, depth = depth, heads = heads, dim_head = latent_dim // heads, ff_expansion=4, attn_dropout=0., ff_dropout=0.)

    speech = torch.randn(32, 128, 512)
    watermark = torch.randn(32, 128, 256)
    pdb.set_trace()
    watermarked_speech = model(speech, watermark)
    pdb.set_trace()
