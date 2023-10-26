import torch
from torch import nn
from dcn_v2 import DCN
import math
from pytorch_wavelets import DWT2D
from torch.cuda import amp
from torch.nn import functional as F

BN_MOMENTUM = 0.1


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUpOnes(nn.Module):
    def __init__(self, startp, end, channels, out_channels=None):
        super(DLAUpOnes, self).__init__()
        self.startp = startp
        self.end = end
        if out_channels is None:
            out_channels = channels
        self.channels = channels
        channels = list(channels)
        for i in range(1, end - startp):
            in_ch = channels[i]
            out_ch = out_channels[i - 1]
            proj = DeformConv(in_ch, out_ch)
            node = DeformConv(out_ch, out_ch)
            up = nn.ConvTranspose2d(out_ch, out_ch, 3, 2, 1, output_padding=1, bias=False, groups=out_ch)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers):
        with amp.autocast(enabled=False):
            for i in range(self.startp, self.end - 1):
                upsample = getattr(self, 'up_' + str(i - self.startp + 1))
                project = getattr(self, 'proj_' + str(i - self.startp + 1))
                node = getattr(self, 'node_' + str(i - self.startp + 1))
                layers[i] = node(layers[i] + upsample(project(layers[i + 1].to(torch.float32))))


class LevelUp(nn.Module):
    def __init__(self, startp, end, in_channels, db, out_channels=None):
        super(LevelUp, self).__init__()
        self.startp = startp
        self.end = end
        self.num = len(in_channels)
        if out_channels is None:
            out_channels = in_channels
        for i in range(1, end - startp):
            layering = LevelSplit(in_channels[i], in_channels[i - 1], out_channels[i - 1], 2, wave=db[i - 1])
            setattr(self, f"level_{i}", layering)

    def forward(self, layers):
        for i in range(0, self.end - self.startp - 1):
            layers[self.startp + i] = self.__getattr__(f"level_{i + 1}")\
                (tuple(layers[self.startp + i: self.startp + i + 2]))


class CombineUp(nn.Module):
    def __init__(self, large_ch, small_ch, out_ch, scale):
        super().__init__()
        self.proj = DeformConv(small_ch, large_ch)
        self.node = DeformConv(large_ch, out_ch)
        self.up = nn.ConvTranspose2d(large_ch, large_ch, scale * 2, stride=scale,
                                padding=scale // 2, output_padding=0,
                                groups=large_ch, bias=False)
        fill_up_weights(self.up)

    def forward(self, large, small):
        small = self.up(self.proj(small))
        return self.node(large + small)


class LevelSplit(nn.Module):
    def __init__(self,
                 in_channel=512,
                 top_channel=256,
                 out_channel=256,
                 level_num=2,
                 scale=2,
                 wave='db4',
                 ):
        super(LevelSplit, self).__init__()
        assert scale >= 2, "scale should be more than 2"
        self.level_num = level_num
        self.out_channel = out_channel
        self.top_channel = top_channel
        self.in_channel = in_channel
        self.dwt = DWT2D(mode='per', J=1, wave=wave)
        if scale == 2:
            upLayer = nn.Sequential()
        else:
            upLayer = nn.ConvTranspose2d(in_channel, in_channel, scale, scale // 2,
                               padding=scale // 4, groups=top_channel, bias=False)
        self.split_conv = nn.Sequential(
            upLayer,
            nn.Conv2d(in_channel, top_channel * level_num, 1)
        )
        self.high_compose_conv = nn.Conv2d(top_channel * 3, top_channel, 3, 1, 1)
        self.compose_conv = nn.Conv2d(top_channel, top_channel, 3, 1, 1)
        self.up_conv = nn.ConvTranspose2d(top_channel, out_channel, 4, 2,
                                          padding=1, groups=out_channel, bias=False)
        self.out_conv = DCN(out_channel, out_channel, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        l1, l2 = x[0].to(torch.float32), x[1]   # top_channel, in_channel
        l2 = self.split_conv(l2)
        l2 = l2.unflatten(1, (self.level_num, -1)).flatten(0, 1).to(torch.float32)  # n * 2, top_channel, h / 2, w / 2
        with amp.autocast(enabled=False):
            dec = self.decompose(l1)    # (n * 2, top_channel, h / 2, w /2)
            up = self.up_conv(dec + l2)  # n * 2, out_channel, h, w
            out = self.out_conv(up)
        return out

    def decompose(self, x):
        low, high = self.dwt(x)
        high = high[0].permute(0, 2, 1, 3, 4).flatten(1, 2)  # n, 3 * top_channel, h, w
        high = self.high_compose_conv(high)[:, None]
        dec = torch.cat((low[:, None], high), dim=1).flatten(0, 1)  # n * 2, top_channel, h, w
        return self.compose_conv(dec)

    @staticmethod
    def orthogonalize(a: torch.Tensor, b: torch.Tensor, gamma=128):
        div = (a ** 2).sum(dim=2, keepdim=True) / gamma
        factor = (a * b).sum(dim=2, keepdim=True) / gamma
        b_ = div * b - a * factor
        b_ = (b_ - b_.mean(dim=2, keepdim=True)) / (b_.std(dim=2, keepdim=True) + 1e-5)    # normalize
        a_ = (a - a.mean(dim=2, keepdim=True)) / (a.std(dim=2, keepdim=True) + 1e-5)
        return a_, b_


class CrossUpdate(nn.Module):
    def __init__(self, in_channels,
                 head_channels=64,
                 radius=3,
                 dilation=2,
                 ):
        '''

        :param in_channels:
        :param radius:
        :param dilation:
        :param mode: [None, 'adaptive']
        :param input_type: ['images', 'video']
        '''

        super(CrossUpdate, self).__init__()
        self.radius = radius
        self.dilation = dilation
        self.template = None
        self.last_template = None
        self.levels = None
        self.x_in = nn.Conv2d(in_channels, head_channels, 3, 1, 1)
        self.x_out = nn.Conv2d(head_channels, in_channels, 3, 1, 1)
        self.x_enhance = StarGRU(head_channels, radius, dilation)
        self.template_update = StarGRU(head_channels, radius, dilation)

    def forward(self, x):
        if self.template is None:
            self.template = self.x_in(x)
            return x
        else:
            x = self.x_in(x)
            x_ = self.x_enhance(self.template, x)
            self.last_template = self.template
            self.template = self.template_update(x, self.template)
            return self.x_out(x_)

    def new_sequence(self):
        self.template = None
        self.last_template = None


class StarGRU(nn.Module):
    def __init__(self, in_channel, radius, dilation):
        super(StarGRU, self).__init__()
        self.radius = radius
        self.dilation = dilation
        # self.z_conv = nn.Conv2d(in_channel * 2, in_channel,3, 1, 1)
        # self.h_conv = nn.Conv2d(in_channel * 2, in_channel,3, 1, 1)
        self.z_conv = DCN(in_channel * 2, in_channel, kernel_size=3, stride=1,
                          padding=1, dilation=1, deformable_groups=1)
        self.h_conv = DCN(in_channel * 2, in_channel, kernel_size=3, stride=1,
                          padding=1, dilation=1, deformable_groups=1)

    def forward(self, x, t):
        tx_ = torch.cat((x, t), dim=1)

        # reset gate
        z = torch.sigmoid(self.z_conv(tx_))

        # update gate
        r, _ = neighbor_correlation_filter(t, x, self.radius, self.dilation)

        _xt_ = torch.cat((r, t), dim=1)
        _xt_ = torch.tanh(self.h_conv(_xt_))

        t = (1. - z) * t + z * _xt_

        return t


class Attribution(nn.Module):
    def __init__(self, in_channels=256,
                 head_channels=128,
                 radius=2,
                 dilation=1
                 ):
        super(Attribution, self).__init__()
        self.levels = None
        self.radius = radius
        self.dilation = dilation

        self.conv1 = nn.Conv2d(in_channels, head_channels, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 1),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(head_channels, in_channels, 1)

    def forward(self, x):
        # self local attention
        x = self.conv1(x)
        attribution, _ = neighbor_correlation_filter(x, x, self.radius, self.dilation)
        attribution = torch.sigmoid(self.conv2(x)) * attribution
        out = self.out(attribution)

        return out


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def neighbor_correlation_filter(f1: torch.Tensor,
                                f2: torch.Tensor,
                                radius=3,
                                dilation=1
                                ):
    '''
        Measure the correlation of channel vectors using vector dot product.
    Args:
        f1: (n, c, h, w) tensor
        f2: (n, c, h, w) tensor
        radius:
        dilation:

    Returns: (n, (2*radius+1)**2, h, w) tensor

    '''

    n, c, h, w = f2.shape

    kernel_size = 2 * radius + 1
    neighbor_nums = kernel_size ** 2
    pad = int((kernel_size - 1) * dilation / 2)

    f2 = F.unfold(f2, kernel_size, padding=pad, dilation=dilation).contiguous()
    f2 = f2.view(n, c, neighbor_nums, h, w)
    # print(f1.shape, f2.shape)

    f1 = f1.view(n, c, 1, h, w)
    corr = torch.sum(f1 * f2, dim=1) / torch.sqrt(torch.tensor(c))
    corr_ = corr.view(n, 1, neighbor_nums, h, w).softmax(dim=2)
    sample = (corr_ * f2).sum(dim=2)

    return sample, corr
