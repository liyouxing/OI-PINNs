""" @ Time: 2025/4/15 20:41  @ Author: Youxing Li  @ Email: 940756344@qq.com """
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from model.NetBlock import UArch, MetaBlock, ConvBlock


# ls-fft uwp-lap estimation from wp
def reflection_full_ts4(ts):
    reflect_top = torch.cat([ts, torch.flip(ts, dims=[-1])], dim=-1)
    reflect_ts = torch.cat([reflect_top, torch.flip(reflect_top, dims=[-2])], dim=-2)

    return reflect_ts


def correct_interval_ts4(ts):
    modified = True
    while modified:
        # 批量修正
        ts = torch.where(ts <= -math.pi, ts + 2 * math.pi, ts)
        ts = torch.where(ts > math.pi, ts - 2 * math.pi, ts)

        # 检查是否还需继续修正
        modified = (torch.max(ts) > math.pi) or (torch.min(ts) <= -math.pi)
    return ts


def row_diff_ts4(psi):
    pad_psi = F.pad(psi, pad=(0, 0, 1, 1), mode='replicate')
    return correct_interval_ts4(pad_psi[:, :, 1:, :] - pad_psi[:, :, :-1, :])


def col_diff_ts4(psi):
    pad_psi = F.pad(psi, pad=(1, 1, 0, 0), mode='replicate')
    return correct_interval_ts4(pad_psi[:, :, :, 1:] - pad_psi[:, :, :, :-1])


def cal_rho_ts4(psi):
    row_dif, col_dif = row_diff_ts4(psi), col_diff_ts4(psi)
    rho = row_dif[:, :, 1:, :] - row_dif[:, :, :-1, :] + col_dif[:, :, :, 1:] - col_dif[:, :, :, :-1]

    return rho


class GradNN(nn.Module):

    def __init__(self, inp_chs=1, oup_chs=1, chs=16, bias=False, padding_mode='replicate',
                 act=nn.GELU):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(inp_chs, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            act())

        self.backbone = UArch(chs=chs, block=MetaBlock, bias=bias, padding_mode=padding_mode, act=act)

        self.end = nn.Conv2d(chs, oup_chs, 1, 1, 0, bias=bias)

    def forward(self, inp):
        feat = self.head(inp)
        grad_feat = self.backbone(feat)
        grad = self.end(grad_feat)

        return grad


def _init_inv_lapF(n, normalization=False):
    kernel = torch.zeros((n, n), dtype=torch.float32)
    kernel[n // 2, n // 2] = -4
    kernel[n // 2 + 1, n // 2] = 1
    kernel[n // 2 - 1, n // 2] = 1
    kernel[n // 2, n // 2 + 1] = 1
    kernel[n // 2, n // 2 - 1] = 1

    k = torch.fft.fft2(kernel)
    k = k.real
    k[0, 0] = 1  # 防止该项为0
    inv_f = (1 / k)

    if normalization:
        inv_f = (inv_f - torch.min(inv_f)) / (torch.max(inv_f) - torch.min(inv_f))
    return inv_f


def _init_inv_random(n):
    kernel = 1 - torch.rand((n, n), dtype=torch.float32)

    return kernel


class PIPUN(nn.Module):

    def __init__(self, inp_chs=1, oup_chs=1, chs=16, ff_size=1024, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()
        self.size = ff_size

        self.gradNN = GradNN(inp_chs=inp_chs, oup_chs=oup_chs, chs=chs, bias=bias,
                             padding_mode=padding_mode, act=act).requires_grad_(requires_grad=False)

        self.inv_head = nn.Sequential(
            nn.Conv2d(1, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            act(),
        )
        # self.register_buffer('inv_lap_filter', _init_inv_lapF(ff_size).unsqueeze(0).unsqueeze(0))
        self.inv_lap_filter = nn.Parameter(_init_inv_lapF(ff_size).unsqueeze(0).unsqueeze(0))
        self.inv_end = nn.Sequential(
            UArch(chs, ConvBlock),
            nn.Conv2d(chs, oup_chs, 3, 1, 1, bias=bias, padding_mode=padding_mode)
        )

    def forward(self, WP):
        _, _, H, W = WP.shape
        # grad_wp = edge_laplacian_ts4(WP)
        # grad by network
        grad_uwp = self.gradNN(WP)

        # inv stage
        F_feat = torch.fft.fft2(self.inv_head(reflection_full_ts4(grad_uwp)),
                                dim=(-2, -1), norm="ortho")  # B, C, H, W; Complex
        F_feat_Re = F_feat.real * self.inv_lap_filter
        F_feat_Im = F_feat.imag * self.inv_lap_filter
        F_feat = torch.view_as_complex(torch.stack(
            [F_feat_Re.unsqueeze(-1), F_feat_Im.unsqueeze(-1)], dim=-1)).squeeze(-1)
        feat = torch.fft.ifftshift(torch.fft.ifft2(F_feat, dim=(-2, -1), norm="ortho"), dim=(-2, -1))[:, :, :H, :W]
        uwp = self.inv_end(feat.real)

        return grad_uwp, uwp


class PIPUN2(nn.Module):
    """ End2EndNN replacing NN2,3, and inv operator """

    def __init__(self, inp_chs=1, oup_chs=1, chs=16, ff_size=1024, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()
        self.size = ff_size

        self.gradNN = GradNN(inp_chs=inp_chs, oup_chs=oup_chs, chs=chs, bias=bias,
                             padding_mode=padding_mode, act=act).requires_grad_(requires_grad=False)

        self.invNN = nn.Sequential(
            nn.Conv2d(1, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            act(),
            UArch(chs),
            nn.Conv2d(chs, oup_chs, 3, 1, 1, bias=bias, padding_mode=padding_mode)
        )

    def forward(self, WP):
        grad_uwp = self.gradNN(WP)
        # inv stage
        uwp = self.invNN(grad_uwp)

        return grad_uwp, uwp


class PIPUN_P(nn.Module):
    """ NN2 NN3, and learned inv operator replace by inv operator"""

    def __init__(self, inp_chs=1, oup_chs=1, chs=16, ff_size=1024, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()
        self.size = ff_size

        self.gradNN = GradNN(inp_chs=inp_chs, oup_chs=oup_chs, chs=chs, bias=bias,
                             padding_mode=padding_mode, act=act).requires_grad_(requires_grad=False)

        self.inv_head = nn.Sequential(
            nn.Conv2d(1, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            act(),
        )
        self.register_buffer('inv_lap_filter', _init_inv_lapF(ff_size).unsqueeze(0).unsqueeze(0))

    def forward(self, WP):
        _, _, H, W = WP.shape
        grad_uwp = self.gradNN(WP)

        # inv stage
        F_feat = torch.fft.fft2(reflection_full_ts4(grad_uwp), dim=(-2, -1), norm="ortho")  # B, 1, H, W; Complex
        F_feat_Re = F_feat.real * self.inv_lap_filter
        F_feat_Im = F_feat.imag * self.inv_lap_filter
        F_feat = torch.view_as_complex(torch.stack(
            [F_feat_Re.unsqueeze(-1), F_feat_Im.unsqueeze(-1)], dim=-1)).squeeze(-1)
        feat = torch.fft.ifftshift(torch.fft.ifft2(F_feat, dim=(-2, -1), norm="ortho"), dim=(-2, -1))[:, :, :H, :W]
        uwp = feat.real

        return grad_uwp, uwp


if __name__ == "__main__":
    import os
    import time
    from ptflops import get_model_complexity_info

    mode = 2
    net = PIPUN()
    if mode == 1:
        # real times on RTX3090
        os.environ['CUDA_VISIBLE_DEVICES'] = "1"
        net = net.cuda().eval()
        total = 0.
        ts = torch.ones([1, 1, 512, 512]).cuda()
        with torch.no_grad():
            for _ in range(1000):
                torch.cuda.synchronize()
                start = time.time()
                result = net(ts)
                torch.cuda.synchronize()
                end = time.time()
                print(end - start)
                total += (end - start)
            print("avg:" + str(total / 100.))
    elif mode == 2:
        num_params = 0
        for k, v in net.named_parameters():
            num_params += v.numel()
        print(num_params)
    else:
        # FLOPs
        macs, params = get_model_complexity_info(net, (1, 512, 512), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
