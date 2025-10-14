""" @ Time: 2025/4/15 15:23  @ Author: Youxing Li  @ Email: 940756344@qq.com
LS/ MLS

Step1: 分离目标区域和背景区域 => 代码：构造背景区域掩码
Step2: 对背景区域进行多项式最小二乘拟合 => 代码：多项式 * 背景区域掩码 -> 待拟合的数据 -> 最小二乘拟合得到多项式对应的系数
Step3: 根据步骤2得到的系数和对应的项相乘得到拟合的畸变相位，与原相位相见 => 代码：
"""
import numpy as np
from scipy.special import factorial
from scipy.optimize import least_squares
from zernpy import ZernPol
import torch
import math


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
    circle: 圆形基底还是方形基底
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


# Zernpy Package基底
def zernike_polynomial_pkg(n, m, size, circle=False):
    """ 生成Zernike多项式
    Params
    -------
    n(int): 阶数
    m(int): 次数
    size(int): 多项式基底大小
    circle: 圆形基底还是方形基底
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

    zp = ZernPol(n=n, m=m)
    zernike_np2d = zp.polynomial_value(rho, theta) * circle_mask
    return zernike_np2d


# LS fitting
def func(items, coef):
    """ 待拟合的函数

    Params
    --------
    items: [poly_1, poly_2, ...], each poly denote a HxW vector
    coef: [coef_1, coef_2, ...], 需要初始化一个值，并且这是被更新的参数
    """
    assert len(coef) == len(items), "coef len should equal to items len!"

    poly = np.zeros_like(items[0])
    for itm_i, coef_i in zip(items, coef):
        poly += itm_i * coef_i

    return poly


def residuals(coef, poly_items, gt):
    return gt - func(poly_items, coef)


def LS_fit(init_coef, poly_items, gt, minimize_func=residuals):
    """ 最小二乘拟合

    Params
    -------
    init_coef: 初始化多项式的系数
    poly_items： 多项式项
    gt: 待拟合的目标
    minimize_func:优化的函数
    """
    plsq = least_squares(minimize_func, init_coef, args=(poly_items, gt))

    opt_coef = plsq

    return opt_coef


def PC_LS_Zernike(abr_phi, mask=None, zern_order=3):
    """
    LS - Zernike
    Params
    --------
    abr_phi: 待校正的相位, 2d numpy
    mask: 背景区域掩码, if mask is None, LS, else MLS
    zern_order: 拟合的多项式介=阶数

    return
    --------
    校正的相位
    """
    poly_coef = []
    poly_items = []

    h, w = abr_phi.shape
    assert h == w, 'h should be equal to w'

    if mask is None:
        mask = 1.

    # init n-order Zernike Poly
    for nds in range(zern_order + 1):
        for mds in range(-1 * nds, nds + 2, 2):
            # plt.imshow(zernike_polynomial(n=nds, m=mds, size=h))
            # plt.show()
            poly_items.append(np.reshape(zernike_polynomial(n=nds, m=mds, size=h) * mask, [h * w]))
            poly_coef.append(0.)  # init ceof

    # LS result
    gt = np.reshape(abr_phi * mask, [h * w])  # reshape H,W -> H*W
    opt_result = LS_fit(init_coef=poly_coef, poly_items=poly_items, gt=gt)
    opt_coef = opt_result['x']

    # AC_phi surface fitting
    ac_phi = np.zeros_like(abr_phi)
    coef_i = 0
    for nds in range(zern_order + 1):
        for mds in range(-1 * nds, nds + 2, 2):
            ac_phi += opt_coef[coef_i] * zernike_polynomial(n=nds, m=mds, size=h)
            coef_i += 1

    # subtracting
    correct_phi = abr_phi - ac_phi

    return correct_phi, opt_coef


def batch_PC_LS_Zernike(abr_phi, mask=None, zern_order=1):
    b, c, h, w = abr_phi.shape
    assert c == 1
    if mask is not None:
        assert mask.shape == abr_phi.shape
        mask = mask.cpu().numpy()
    else:
        mask = np.ones([b, c, h, w])
        mask[:, :, 25:490, :] = 0.

    correct_phis = torch.zeros_like(abr_phi)
    for i in range(b):
        correct_phi, _ = PC_LS_Zernike(abr_phi[i, 0].cpu().numpy(), mask=mask[i, 0], zern_order=zern_order)
        correct_phis[i, 0] = torch.from_numpy(correct_phi).cuda()
    return correct_phis


# pytorch mls

def radial_zernike_poly(n, m, r):
    """
    计算 Zernike 径向多项式 R_n^m(r)
    """
    R = torch.zeros_like(r)
    for s in range((n - m) // 2 + 1):
        c = ((-1) ** s * math.factorial(n - s) /
             (math.factorial(s) *
              math.factorial((n + m) // 2 - s) *
              math.factorial((n - m) // 2 - s)))
        R += c * r.pow(n - 2 * s)
    return R


def calculate_zernike_basis(H, W, max_n):
    """
    构建 Zernike 基的张量 B，大小为 (M, H, W)
    """
    y = torch.linspace(-1, 1, H).unsqueeze(1).expand(H, W)
    x = torch.linspace(-1, 1, W).unsqueeze(0).expand(H, W)
    r = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(y, x)

    zernike_modes = []
    for n in range(max_n + 1):
        for m in range(-n, n + 1, 2):
            if m == 0:
                Z = radial_zernike_poly(n, m, r)
            elif m > 0:
                Z = radial_zernike_poly(n, m, r) * torch.cos(m * theta)
            else:
                Z = radial_zernike_poly(n, -m, r) * torch.sin(-m * theta)
            zernike_modes.append(Z)
    B = torch.stack(zernike_modes, dim=0)  # (M, H, W)
    return B


def zernike_fit_and_correct(phase_distorted, mask=None, max_n=10):
    """
    通过最小二乘法拟合 Zernike 多项式并修正相位
    """
    H, W = phase_distorted.shape

    if mask is None:
        mask = 1.

    B = calculate_zernike_basis(H, W, max_n)  # (M, H, W)
    M = B.shape[0]
    # 展平
    MB_flat = (B * mask).view(M, -1).T  # (H*W, M)
    B_flat = B.view(M, -1).T  # (H*W, M)

    M_phase_flat = (phase_distorted * mask).contiguous().view(-1, 1)  # (H*W, 1)

    # 使用 Moore-Penrose 伪逆求解最小二乘问题
    B_pinv = torch.pinverse(MB_flat)  # (M, H*W)
    coeffs = B_pinv @ M_phase_flat  # (M, 1)

    # 重建背景相位
    bg_flat = B_flat @ coeffs  # (H*W, 1)
    bg_phase = bg_flat.view(H, W)  # (H, W)

    # 校正相位
    corrected_phase = phase_distorted - bg_phase

    return corrected_phase, coeffs.squeeze(), bg_phase


if __name__ == "__main__":
    pass
