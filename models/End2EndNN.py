""" @ Time: 2025/8/1 14:47  @ Author: Youxing Li  @ Email: 940756344@qq.com """
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, chs, bias=False, padding_mode='replicate', act=nn.ReLU, norm=nn.Identity):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(chs, chs, 5, 1, 2, bias=bias, padding_mode=padding_mode),
            norm(chs),
            act(),
        )

    def forward(self, x):
        return self.conv(x)


class UArch(nn.Module):
    def __init__(self, chs, block=ConvBlock, bias=False, padding_mode='replicate', act=nn.ReLU, norm=nn.Identity):
        super().__init__()

        self.init_block = nn.Sequential(
            block(chs),
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(chs, chs * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            norm(chs * 2),
            act(),
            block(chs * 2, norm=norm),
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(chs * 2, chs * 4, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            norm(chs * 4),
            act(),
            block(chs * 4, norm=norm),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(chs * 4, chs * 8, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            norm(chs * 8),
            act(),
            block(chs * 8, norm=norm),
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(chs * 8, chs * 16, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            norm(chs * 16),
            act(),
        )

        self.block = nn.Sequential(
            block(chs * 16, norm=norm),
            block(chs * 16, norm=norm),  # PU
            block(chs * 16, norm=norm),
            block(chs * 16, norm=norm),
            block(chs * 16, norm=norm),
            block(chs * 16, norm=norm),
            block(chs * 16, norm=norm),
            block(chs * 16, norm=norm),

        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(chs * 16, chs * 8, 2, 2, 0, bias=bias),
            norm(chs * 8),
            act(),
        )

        self.dec4 = nn.Sequential(
            nn.Conv2d(chs * 16, chs * 8, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            norm(chs * 8),
            act(),
            block(chs * 8, norm=norm),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(chs * 8, chs * 4, 2, 2, 0, bias=bias),
            norm(chs * 4),
            act(),
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(chs * 8, chs * 4, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            norm(chs * 4),
            act(),
            block(chs * 4, norm=norm),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(chs * 4, chs * 2, 2, 2, 0, bias=bias),
            norm(chs * 2),
            act(),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(chs * 4, chs * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            norm(chs * 2),
            act(),
            block(chs * 2, norm=norm),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(chs * 2, chs, 2, 2, 0, bias=bias),
            norm(chs),
            act(),
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(chs * 2, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            norm(chs),
            act(),
            block(chs, norm=norm),
        )

    def forward(self, feats):
        x1 = self.init_block(feats)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y5 = self.block(x5)
        y4 = self.dec4(torch.cat([self.up4(y5), x4], 1))
        y3 = self.dec3(torch.cat([self.up3(y4), x3], 1))
        y2 = self.dec2(torch.cat([self.up2(y3), x2], 1))
        oup_feat = self.dec1(torch.cat([self.up1(y2), x1], 1))

        return oup_feat


class End2EndNN(nn.Module):
    def __init__(self, inp_chs=1, oup_chs=1, chs=16, bias=False, padding_mode='replicate',
                 act=nn.ReLU, norm=nn.BatchNorm2d):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(inp_chs, chs, 3, 1, 1, bias=bias, padding_mode=padding_mode),
            norm(chs),
            act())

        self.backbone = UArch(chs=chs, bias=bias, padding_mode=padding_mode, act=act, norm=norm)

        self.end = nn.Conv2d(chs, oup_chs, 1, 1, 0, bias=bias)

    def forward(self, inp):
        oup = self.end(self.backbone(self.head(inp)))

        return oup


if __name__ == "__main__":
    import os
    import time
    from ptflops import get_model_complexity_info

    mode = 3
    net = End2EndNN()
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
