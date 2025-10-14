""" @ Time: 2025/4/15 20:41  @ Author: Youxing Li  @ Email: 940756344@qq.com """
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.NetBlock import ResBlock, UArch


# PR FreqNet
def win_init1_ts2(data_size, win_center=(0, 0), win_size=None, mode="row",
                  init_type=torch.hann_window, k_beta=0.1):
    """
        window init in row or column

        Params
        --------
        data_size: data size
        win_center: (0, 0) -> in the center
        win_size: freq win size
        mode: 'row' | 'column'
        init_type: torch.hann_window | torch.hamming_window | torch.blackman_window |
                   torch.kaiser_window | torch.bartlett_window
        k_beta: kaiser beta
        """
    assert mode in {'row', 'column'}, 'mode error'
    assert init_type in {torch.hann_window, torch.hamming_window, torch.blackman_window,
                         torch.kaiser_window, torch.bartlett_window}, 'init type error'
    if win_size is None:
        win_size = data_size
    assert (win_size[0] <= data_size[0]) and (win_size[1] <= data_size[1]), 'freq win size error'

    if mode == "row":  # win init in row
        if init_type in {torch.hann_window, torch.hamming_window, torch.blackman_window, torch.bartlett_window}:
            freq_win = init_type(win_size[1], periodic=False).unsqueeze(0).repeat(win_size[0], 1)
        else:  # kaiser
            freq_win = init_type(win_size[1], beta=k_beta, periodic=False).unsqueeze(0).repeat(win_size[0], 1)
    else:  # han init in column
        if init_type in {torch.hann_window, torch.hamming_window, torch.blackman_window, torch.bartlett_window}:
            freq_win = init_type(win_size[0], periodic=False).unsqueeze(1).repeat(1, win_size[1])
        else:
            freq_win = init_type(win_size[0], beta=k_beta, periodic=False).unsqueeze(1).repeat(1, win_size[1])

    # padding
    pad_c1 = (data_size[0] - win_size[0]) // 2 - win_center[0]
    pad_c2 = data_size[0] - win_size[0] - pad_c1
    pad_r1 = (data_size[1] - win_size[1]) // 2 + win_center[1]
    pad_r2 = data_size[1] - win_size[1] - pad_r1

    freq_mask = F.pad(freq_win, (pad_r1, pad_r2, pad_c1, pad_c2))

    return freq_mask.detach()


def win_init2_ts2(data_size, win_center=(0, 0), win_size=None,
                  init_type=torch.hann_window, k_beta=0.1):
    """  win init in two dim """
    assert init_type in {torch.hann_window, torch.hamming_window, torch.blackman_window,
                         torch.kaiser_window, torch.bartlett_window}, 'init type error'
    if win_size is None:
        win_size = data_size
    assert (win_size[0] <= data_size[0]) and (win_size[1] <= data_size[1]), 'freq win size error'

    if init_type in {torch.hann_window, torch.hamming_window, torch.blackman_window, torch.bartlett_window}:
        freq_win = torch.outer(init_type(win_size[0], periodic=False), init_type(win_size[1], periodic=False))
    else:  # kaiser
        freq_win = torch.outer(init_type(win_size[0], periodic=False, beta=k_beta),
                               init_type(win_size[1], periodic=False, beta=k_beta))

    pad_c1 = (data_size[0] - win_size[0]) // 2 + win_center[0]
    pad_c2 = data_size[0] - win_size[0] - pad_c1
    pad_r1 = (data_size[1] - win_size[1]) // 2 + win_center[1]
    pad_r2 = data_size[1] - win_size[1] - pad_r1

    freq_mask = F.pad(freq_win, (pad_r1, pad_r2, pad_c1, pad_c2))

    return freq_mask


class PIPDN(nn.Module):
    def __init__(self, inp_chs=1, oup_chs=1, chs=16, ff_size=512, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()

        self.H = ff_size
        self.W = self.H

        self.net_head = nn.Sequential(
            nn.Conv2d(inp_chs, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            act(),
            UArch(chs=chs, block=ResBlock)
        )

        self.filter1 = nn.Parameter(torch.rand((self.H, self.W), dtype=torch.float32).unsqueeze(0))
        """self._filter_init1(data_size=(self.H, self.W), win_center=(0, 0), win_size=(self.H, self.W // 12),
                               mode='row', init_type=torch.hann_window))"""

        self.filter2 = nn.Parameter(torch.rand((self.H, self.W), dtype=torch.float32).unsqueeze(0))
        """self._filter_init2(data_size=(self.H, self.W), win_center=(0, self.W // 12),
                               win_size=(self.H // 12, self.W // 12),
                               init_type=torch.hann_window)"""

        self.sin_end = nn.Sequential(
            UArch(chs=chs, block=ResBlock),
            nn.Conv2d(chs, oup_chs, 1, 1, 0, bias=bias, padding_mode=padding_mode)
        )

        self.cos_end = nn.Sequential(
            UArch(chs=chs, block=ResBlock),
            nn.Conv2d(chs, oup_chs, 1, 1, 0, bias=bias, padding_mode=padding_mode)
        )

    def _filter_init1(self, data_size, win_center, win_size, mode, init_type):
        return (1. - win_init1_ts2(data_size=data_size, win_center=win_center, win_size=win_size,
                                   mode=mode, init_type=init_type)).unsqueeze(0)

    def _filter_init2(self, data_size, win_center, win_size, init_type):
        return win_init2_ts2(data_size=data_size, win_center=win_center,
                             win_size=win_size, init_type=init_type).unsqueeze(0)

    def forward(self, SI):
        _, _, h, w = SI.shape

        # Spatial Domain
        feat = self.net_head(SI)  # B, C, H, W

        # Freq Domain
        F_feat = torch.fft.fftshift(torch.fft.fft2(feat, dim=(2, 3), norm="ortho"), dim=(2, 3))  # B, C, H, W; Complex
        F_feat_Re = F_feat.real * self.filter1 * self.filter2
        F_feat_Im = F_feat.imag * self.filter1 * self.filter2
        F_feat = torch.view_as_complex(torch.stack(
            [F_feat_Re.unsqueeze(-1), F_feat_Im.unsqueeze(-1)], dim=-1)).squeeze(-1)
        feat = torch.fft.ifft2(torch.fft.ifftshift(F_feat, dim=(2, 3)), dim=(2, 3), norm="ortho")  # -> B, C, H, W

        angle_re = self.cos_end(feat.real)  # cos phi
        angle_im = self.sin_end(feat.imag)  # sin phi

        # cos_phi = angle_re / torch.sqrt(angle_re ** 2 + angle_im ** 2)
        # sin_phi = angle_im / torch.sqrt(angle_re ** 2 + angle_im ** 2)

        # Spatial Domain
        WP = torch.atan2(angle_im, angle_re)

        return WP


class PIPDN2(nn.Module):
    """ w/o learnable filters """

    def __init__(self, inp_chs=1, oup_chs=1, chs=16, ff_size=512, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()

        self.H = ff_size
        self.W = self.H

        self.net_head = nn.Sequential(
            nn.Conv2d(inp_chs, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            act(),
            UArch(chs=chs, block=ResBlock)
        )

        self.sin_end = nn.Sequential(
            UArch(chs=chs, block=ResBlock),
            nn.Conv2d(chs, oup_chs, 1, 1, 0, bias=bias, padding_mode=padding_mode)
        )

        self.cos_end = nn.Sequential(
            UArch(chs=chs, block=ResBlock),
            nn.Conv2d(chs, oup_chs, 1, 1, 0, bias=bias, padding_mode=padding_mode)
        )

    def forward(self, SI):
        _, _, h, w = SI.shape

        # Spatial Domain
        feat = self.net_head(SI)  # B, C, H, W

        angle_re = self.cos_end(feat)  # cos phi
        angle_im = self.sin_end(feat)  # sin phi

        # Spatial Domain
        WP = torch.atan2(angle_im, angle_re)

        return WP


class PIPDN3(nn.Module):
    """ replace NN1 by conv"""

    def __init__(self, inp_chs=1, oup_chs=1, chs=16, ff_size=512, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()

        self.H = ff_size
        self.W = self.H

        self.net_head = nn.Sequential(
            nn.Conv2d(inp_chs, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            act(),
        )

        self.filter1 = nn.Parameter(
            self._filter_init1(data_size=(self.H, self.W), win_center=(0, 0), win_size=(self.H, self.W // 12),
                               mode='row', init_type=torch.hann_window))

        self.filter2 = nn.Parameter(self._filter_init2(data_size=(self.H, self.W), win_center=(0, self.W // 12),
                                                       win_size=(self.H // 12, self.W // 12),
                                                       init_type=torch.hann_window))

        self.sin_end = nn.Sequential(
            UArch(chs=chs, block=ResBlock),
            nn.Conv2d(chs, oup_chs, 1, 1, 0, bias=bias, padding_mode=padding_mode)
        )

        self.cos_end = nn.Sequential(
            UArch(chs=chs, block=ResBlock),
            nn.Conv2d(chs, oup_chs, 1, 1, 0, bias=bias, padding_mode=padding_mode)
        )

    def _filter_init1(self, data_size, win_center, win_size, mode, init_type):
        return (1. - win_init1_ts2(data_size=data_size, win_center=win_center, win_size=win_size,
                                   mode=mode, init_type=init_type)).unsqueeze(0)

    def _filter_init2(self, data_size, win_center, win_size, init_type):
        return win_init2_ts2(data_size=data_size, win_center=win_center,
                             win_size=win_size, init_type=init_type).unsqueeze(0)

    def forward(self, SI):
        _, _, h, w = SI.shape

        # Spatial Domain
        feat = self.net_head(SI)  # B, C, H, W

        # Freq Domain
        F_feat = torch.fft.fftshift(torch.fft.fft2(feat, dim=(2, 3), norm="ortho"), dim=(2, 3))  # B, C, H, W; Complex
        F_feat_Re = F_feat.real * self.filter1 * self.filter2
        F_feat_Im = F_feat.imag * self.filter1 * self.filter2
        F_feat = torch.view_as_complex(torch.stack(
            [F_feat_Re.unsqueeze(-1), F_feat_Im.unsqueeze(-1)], dim=-1)).squeeze(-1)
        feat = torch.fft.ifft2(torch.fft.ifftshift(F_feat, dim=(2, 3)), dim=(2, 3), norm="ortho")  # -> B, C, H, W

        angle_re = self.cos_end(feat.real)  # cos phi
        angle_im = self.sin_end(feat.imag)  # sin phi

        # Spatial Domain
        WP = torch.atan2(angle_im, angle_re)

        return WP


class PIPDN4(nn.Module):
    """ replace NN2 and NN3 by conv"""

    def __init__(self, inp_chs=1, oup_chs=1, chs=16, ff_size=512, bias=False, padding_mode='replicate', act=nn.GELU):
        super().__init__()

        self.H = ff_size
        self.W = self.H

        self.net_head = nn.Sequential(
            nn.Conv2d(inp_chs, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            act(),
            UArch(chs=chs, block=ResBlock)
        )

        self.filter1 = nn.Parameter(
            self._filter_init1(data_size=(self.H, self.W), win_center=(0, 0), win_size=(self.H, self.W // 12),
                               mode='row', init_type=torch.hann_window))

        self.filter2 = nn.Parameter(self._filter_init2(data_size=(self.H, self.W), win_center=(0, self.W // 12),
                                                       win_size=(self.H // 12, self.W // 12),
                                                       init_type=torch.hann_window))

        self.sin_end = nn.Sequential(
            nn.Conv2d(chs, oup_chs, 1, 1, 0, bias=bias, padding_mode=padding_mode)
        )

        self.cos_end = nn.Sequential(
            nn.Conv2d(chs, oup_chs, 1, 1, 0, bias=bias, padding_mode=padding_mode)
        )

    def _filter_init1(self, data_size, win_center, win_size, mode, init_type):
        return (1. - win_init1_ts2(data_size=data_size, win_center=win_center, win_size=win_size,
                                   mode=mode, init_type=init_type)).unsqueeze(0)

    def _filter_init2(self, data_size, win_center, win_size, init_type):
        return win_init2_ts2(data_size=data_size, win_center=win_center,
                             win_size=win_size, init_type=init_type).unsqueeze(0)

    def forward(self, SI):
        _, _, h, w = SI.shape

        # Spatial Domain
        feat = self.net_head(SI)  # B, C, H, W

        # Freq Domain
        F_feat = torch.fft.fftshift(torch.fft.fft2(feat, dim=(2, 3), norm="ortho"), dim=(2, 3))  # B, C, H, W; Complex
        F_feat_Re = F_feat.real * self.filter1 * self.filter2
        F_feat_Im = F_feat.imag * self.filter1 * self.filter2
        F_feat = torch.view_as_complex(torch.stack(
            [F_feat_Re.unsqueeze(-1), F_feat_Im.unsqueeze(-1)], dim=-1)).squeeze(-1)
        feat = torch.fft.ifft2(torch.fft.ifftshift(F_feat, dim=(2, 3)), dim=(2, 3), norm="ortho")  # -> B, C, H, W

        angle_re = self.cos_end(feat.real)  # cos phi
        angle_im = self.sin_end(feat.imag)  # sin phi

        # cos_phi = angle_re / torch.sqrt(angle_re ** 2 + angle_im ** 2)
        # sin_phi = angle_im / torch.sqrt(angle_re ** 2 + angle_im ** 2)

        # Spatial Domain
        WP = torch.atan2(angle_im, angle_re)

        return WP


if __name__ == "__main__":
    import os
    import time
    from ptflops import get_model_complexity_info

    mode = 2
    net = PIPDN()
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
