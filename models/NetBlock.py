""" @ Time: 2025/4/15 20:57  @ Author: Youxing Li  @ Email: 940756344@qq.com
Ref:
Restormer: CVPR2022
PoolFormer: CVPR2022
SAFMN: ICCV2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# basic conv block
class ConvBlock(nn.Module):
    def __init__(self, chs, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(chs, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            act(),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, chs, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(chs, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            act(),
            nn.Conv2d(chs, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
        )

        self.act = act()

    def forward(self, x):
        return self.act(self.conv(x) + x)


# Restormer
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2,
                                    kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim,
                                     kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, conv_boost):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.rp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        ) if conv_boost else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out) + self.rp(x)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, head_dim=16, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                 conv_boost=False):
        super(TransformerBlock, self).__init__()
        if head_dim is None:
            head_dim = dim
        num_heads = dim // head_dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, conv_boost)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


# PoolFormer & SFAMN
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


class Pooling(nn.Module):
    def __init__(self, pool_size=5):
        super().__init__()

        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


# Separable convolution
class SepConv(nn.Module):
    "Inverted separable convolution from MobileNetV2"

    def __init__(self, dim, kernel_size=7, padding=3, expansion_ratio=2,
                 act1=nn.GELU, act2=nn.Identity, bias=False):
        super().__init__()

        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, 1, 1, 0, bias=bias)
        self.act1 = act1()
        self.dwconv = nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size,
                                padding=padding, groups=med_channels, bias=bias)
        self.act2 = act2()
        self.pwconv2 = nn.Conv2d(med_channels, dim, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class MetaBlock(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias', pooling=False):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type=LayerNorm_type)

        # Pooling layer
        self.pool = Pooling() if pooling else SepConv(dim)
        # Feedforward layer
        self.ccm = CCM(dim)

    def forward(self, x):
        x = self.pool(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class UArch(nn.Module):
    def __init__(self, chs, block, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()

        self.init_block = nn.Sequential(
            block(chs),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(chs, chs * 2, 3, 2, 1, bias=bias, padding_mode=padding_mode),
            act(),
            block(chs * 2),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(chs * 2, chs * 4, 3, 2, 1, bias=bias, padding_mode=padding_mode),
            act(),
            block(chs * 4),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(chs * 4, chs * 8, 3, 2, 1, bias=bias, padding_mode=padding_mode),
            act(),
        )

        self.block = nn.Sequential(
            block(chs * 8),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(chs * 8, chs * 4, 3, 2, 1, output_padding=1, bias=bias),
            act(),
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(chs * 8, chs * 4, 1, 1, 0, bias=bias, padding_mode=padding_mode),
            act(),
            block(chs * 4),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(chs * 4, chs * 2, 3, 2, 1, output_padding=1, bias=bias),
            act(),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(chs * 4, chs * 2, 1, 1, 0, bias=bias, padding_mode=padding_mode),
            act(),
            block(chs * 2),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(chs * 2, chs, 3, 2, 1, output_padding=1, bias=bias),
            act(),
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(chs * 2, chs, 1, 1, 0, bias=bias, padding_mode=padding_mode),
            act(),
            block(chs),
        )

    def forward(self, feats):
        x1 = self.init_block(feats)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y4 = self.block(x4)

        y3 = self.dec3(torch.cat([self.up3(y4), x3], 1))
        y2 = self.dec2(torch.cat([self.up2(y3), x2], 1))
        oup_feat = self.dec1(torch.cat([self.up1(y2), x1], 1))

        return oup_feat


class EncoderArch(nn.Module):
    def __init__(self, chs, down_n=5, block=ResBlock, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()
        self.down_n = down_n
        self.init_block = nn.Sequential(
            block(chs),
        )

        self.down_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(chs * (2 ** i), chs * (2 ** (i + 1)), 3, 2, 1, bias=bias, padding_mode=padding_mode),
                act(),
                block(chs * (2 ** (i + 1))),
            ) for i in range(self.down_n)]
        )

    def forward(self, feats):  # (B, C, H, W)
        feats = self.init_block(feats)

        for i in range(self.down_n):
            feats = self.down_layers[i](feats)

        return feats  # (B, 2**self.down_n * C, H//(2**self.down_n), W//(2**self.down_n))
