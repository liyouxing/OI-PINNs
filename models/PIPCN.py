""" @ Time: 2025/4/15 20:41  @ Author: Youxing Li  @ Email: 940756344@qq.com """
import numpy as np
import torch
import torch.nn as nn
from scipy.special import factorial
import torch.nn.functional as F
from model.SegFormer import Segformer, MiT
from model.TradPC import batch_PC_LS_Zernike


# 自定义的Zernike基底
def _zernike_radial(n, m, rho):
    """ 计算Zernike径向多项式
    Params
    --------
    n: Radial order
    m: Angular Frequency
    n和m是含零正整数，并且n-m≥0且为偶数。参数n表示多项式最高阶，它表示多项式径向度数或者阶数，m可以称作方位角频率。
    """
    assert n >= m, 'n should large than m'
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)

    R = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        num = (-1) ** s * factorial(n - s)
        denom = factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s)
        R += num / denom * rho ** (n - 2 * s)
    return R


def zernike_polynomial(n, m, size, circle=False):
    """ 生成Zernike多项式
    Params
    -------
    n(int): 阶数
    m(int): 次数
    size(int): 多项式基底大小
    circle: 圆形还是方形项
    返回:
    numpy.ndarray:计算得到的Zernike多项式值
    """
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)

    rho = np.sqrt(X ** 2 + Y ** 2)  # 极坐标径向值
    circle_mask = (rho <= 1) if circle else 1
    rho = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))

    theta = np.arctan2(Y, X)  # 极坐标角度值，相位会落在 -pi 到 pi
    theta = theta % (2 * np.pi)  # 变到 0 到 2pi

    if m == 0:
        return np.sqrt(n + 1) * _zernike_radial(n, m, rho) * circle_mask
    else:
        radial = _zernike_radial(n, abs(m), rho)
        if m > 0:
            return np.sqrt(2 * (n + 1)) * radial * np.cos(m * theta) * circle_mask
        else:
            return np.sqrt(2 * (n + 1)) * radial * np.sin(-m * theta) * circle_mask


class MaskNN(nn.Module):

    def __init__(self, inp_chs=1, num_classes=2, pad_stride=4):
        super().__init__()
        self.pad_stride = pad_stride
        self.MaskNN = Segformer(dims=(32, 64, 160, 256),
                                channels=inp_chs,
                                decoder_dim=256,
                                num_classes=num_classes,
                                pad_stride=pad_stride)

    def forward(self, phi):
        mask_logits = self.MaskNN(phi)  # (B, 1, H, W)
        mask_logits = F.interpolate(mask_logits, scale_factor=self.pad_stride, mode='nearest')
        return mask_logits


class PIPCN(nn.Module):

    def __init__(self, inp_chs=1, oup_chs=1, zern_order=3, zern_size=512, bias=False):
        super().__init__()
        self.coef_n = 0  # zern coef num
        self.register_buffer('Zern_eigen', self._zern_init(
            order=zern_order, size=zern_size, circle=False, normalization=False))  # (1, coef_n, z_size, z_size)

        self.MaskNN = MaskNN(inp_chs=inp_chs, num_classes=2, pad_stride=4).requires_grad_(requires_grad=False)

        self.ZNN = nn.Sequential(
            MiT(channels=1,
                dims=(32, 64, 160, 256),
                heads=(1, 2, 5, 8),
                ff_expansion=(8, 8, 4, 4),
                reduction_ratio=(8, 4, 2, 1),
                num_layers=(2, 2, 2, 2),
                pad_stride=4),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(64, self.coef_n, 1, 1, 0, bias=bias)
        )

    def _zern_init(self, order, size, circle=False, normalization=False):
        zp = []
        # init n-order Zernike Poly
        for nds in range(order + 1):
            for mds in range(-1 * nds, nds + 2, 2):
                zn = zernike_polynomial(nds, mds, size, circle)
                if normalization:
                    zn = (zn - np.min(zn)) / (np.max(zn) - np.min(zn))

                zp.append(torch.tensor(zn, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
                self.coef_n += 1

        return torch.cat(zp, dim=1)

    def forward(self, phi):
        mask_logits = self.MaskNN(phi)  # (B, 1, H, W)
        mask = torch.argmax(torch.softmax(mask_logits, dim=1), dim=1, keepdim=True)  # obj mask
        mask_phi = phi * (1 - mask)
        coef = self.ZNN(mask_phi)  # (B, coef_n, 1, 1)
        phi_a = torch.sum(coef * self.Zern_eigen, dim=1, keepdim=True)  # (B, 1, zn_size, zn_size)
        phi_o = phi - phi_a

        return phi_o, mask


class PIPCN2(nn.Module):
    """ removing MaskNN """

    def __init__(self, inp_chs=1, oup_chs=1, zern_order=3, zern_size=512, bias=False):
        super().__init__()
        self.coef_n = 0  # zern coef num
        self.register_buffer('Zern_eigen', self._zern_init(
            order=zern_order, size=zern_size, circle=False, normalization=False))  # (1, coef_n, z_size, z_size)

        self.ZNN = nn.Sequential(
            MiT(channels=1,
                dims=(32, 64, 160, 256),
                heads=(1, 2, 5, 8),
                ff_expansion=(8, 8, 4, 4),
                reduction_ratio=(8, 4, 2, 1),
                num_layers=(2, 2, 2, 2),
                pad_stride=4),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(64, self.coef_n, 1, 1, 0, bias=bias)
        )

    def _zern_init(self, order, size, circle=False, normalization=False):
        zp = []
        # init n-order Zernike Poly
        for nds in range(order + 1):
            for mds in range(-1 * nds, nds + 2, 2):
                zn = zernike_polynomial(nds, mds, size, circle)
                if normalization:
                    zn = (zn - np.min(zn)) / (np.max(zn) - np.min(zn))

                zp.append(torch.tensor(zn, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
                self.coef_n += 1

        return torch.cat(zp, dim=1)

    def forward(self, phi):
        mask = torch.zeros_like(phi)
        mask_phi = phi * (1 - mask)
        coef = self.ZNN(mask_phi)  # (B, coef_n, 1, 1)
        phi_a = torch.sum(coef * self.Zern_eigen, dim=1, keepdim=True)  # (B, 1, zn_size, zn_size)
        phi_o = phi - phi_a

        return phi_o, mask


class PIPCN3(nn.Module):
    """ removing Zernike items, direct predict distortion """

    def __init__(self, inp_chs=1, oup_chs=1, zern_order=3, zern_size=512, bias=False):
        super().__init__()

        self.MaskNN = MaskNN(inp_chs=inp_chs, num_classes=2, pad_stride=4).requires_grad_(requires_grad=False)

        self.ZNN = MaskNN(inp_chs=inp_chs, num_classes=1, pad_stride=4)

    def forward(self, phi):
        mask_logits = self.MaskNN(phi)  # (B, 1, H, W)
        mask = torch.argmax(torch.softmax(mask_logits, dim=1), dim=1, keepdim=True)  # obj mask
        mask_phi = phi * (1 - mask)
        phi_a = self.ZNN(mask_phi)
        phi_o = phi - phi_a

        return phi_o, mask


class PIPCN_P(nn.Module):

    def __init__(self, inp_chs=1, oup_chs=1, zern_order=3, zern_size=512, bias=False):
        super().__init__()
        self.coef_n = 0  # zern coef num
        self.register_buffer('Zern_eigen', self._zern_init(
            order=zern_order, size=zern_size, circle=False, normalization=False))  # (1, coef_n, z_size, z_size)

        self.MaskNN = MaskNN(inp_chs=inp_chs, num_classes=2, pad_stride=4).requires_grad_(requires_grad=False)

        self.ZNN = nn.Sequential(
            MiT(channels=1,
                dims=(32, 64, 160, 256),
                heads=(1, 2, 5, 8),
                ff_expansion=(8, 8, 4, 4),
                reduction_ratio=(8, 4, 2, 1),
                num_layers=(2, 2, 2, 2),
                pad_stride=4),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(64, self.coef_n, 1, 1, 0, bias=bias)
        )

    def _zern_init(self, order, size, circle=False, normalization=False):
        zp = []
        # init n-order Zernike Poly
        for nds in range(order + 1):
            for mds in range(-1 * nds, nds + 2, 2):
                zn = zernike_polynomial(nds, mds, size, circle)
                if normalization:
                    zn = (zn - np.min(zn)) / (np.max(zn) - np.min(zn))

                zp.append(torch.tensor(zn, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
                self.coef_n += 1

        return torch.cat(zp, dim=1)

    def forward(self, phi):
        mask_logits = self.MaskNN(phi)  # (B, 1, H, W)
        mask = torch.argmax(torch.softmax(mask_logits, dim=1), dim=1, keepdim=True)  # obj mask
        mask_phi = phi * (1 - mask)
        phi_o = batch_PC_LS_Zernike(abr_phi=phi, mask=mask_phi, zern_order=3)

        return phi_o, mask


if __name__ == "__main__":
    import os
    import time
    from ptflops import get_model_complexity_info

    mode = 2
    net = PIPCN()
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
