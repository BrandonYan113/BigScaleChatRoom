from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
from os.path import join
from pytorch_wavelets import DWT2D
from src.lib.models.networks.subnets import IDAUp, DLAUpOnes, DeformConv, \
    fill_fc_weights, fill_up_weights, LevelSplit, LevelUp, Attribution, CrossUpdate
import numpy as np
import torch
from src.lib.models.networks.memory import MemoryNet
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if hash.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=False, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86.pth')
    return model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LevelIDAUp(nn.Module):
    def __init__(self, out_channels, channels, up_f, wave):
        super(LevelIDAUp, self).__init__()
        self.dwt = DWT2D(mode='per', J=1, wave=wave)

        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, out_channels)
            node = DeformConv(out_channels, out_channels)

            up = nn.ConvTranspose2d(out_channels, 2 * out_channels, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=out_channels, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers):
        compose = self.decompose(layers[0])

    def decompose(self, x):
        low, high = self.dwt(x)
        high = high[0].permute(0, 2, 1, 3, 4).sum(dim=1, keepdim=True)    # n, 1, top_channel, h, w
        dec = torch.cat((low[:, None], high), dim=1).flatten(0, 1)  # n * 2, top_channel, h, w
        return dec


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]

        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        self.heads = heads
        set_attribute(self, head_conv, channels, final_kernel)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        y = []

        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]


class ArchLarge(nn.Module):
    def __init__(self, heads, down_ratio, final_kernel, head_conv, out_channel=0):
        super(ArchLarge, self).__init__()
        assert down_ratio in [4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.base = dla34()
        channels = self.base.channels
        if out_channel == 0:
            out_channel = channels[self.first_level]
        dbs = ["db10", "db8", "db6", "db4", "db3", "db2"]
        # for i in range(self.first_level)
        self.local_attention = Attribution(channels[-1], 256)
        self.association = CrossUpdate(256)

        self.dla_up = DLAUpOnes(self.first_level, len(channels), channels[self.first_level:],
                                channels[self.first_level:])
        self.level_up = LevelUp(self.first_level, len(channels) - 1,
                                channels[self.first_level: -1],
                                dbs[self.first_level: -1])
        self.dla_up2 = DLAUpOnes(self.first_level, len(channels) - 2, channels[self.first_level:-2])
        self.out_conv = nn.Sequential(
            nn.InstanceNorm2d(channels[self.first_level]),
            nn.Conv2d(channels[self.first_level], out_channel, 3, 1, 1)
        )
        self.heads = heads
        set_attribute(self, head_conv, channels, final_kernel)

    def forward(self, x, input_type="image", splen=5):
        x = self.base(x)
        x[-1] = self.local_attention(x[-1])
        self.dla_up(x)
        if input_type == "video":
            sp_imgs = x[-2].unflatten(0, (splen, -1))  # splen, bs, flen, h, w
            spimgs_ = []
            self.association.new_sequence()
            for imgs in sp_imgs:
                spimgs_.append(self.association(imgs))

            spimgs_ = torch.cat(spimgs_, dim=0)
            x[-2] = spimgs_
        self.level_up(x)
        self.dla_up2(x)
        bkout = self.out_conv(x[self.first_level])
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(bkout)
        return [z]


class ArchMiddle(nn.Module):
    def __init__(self, heads, down_ratio, final_kernel, head_conv, out_channel=0):
        super(ArchMiddle, self).__init__()
        assert down_ratio in [4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.base = dla34()
        channels = self.base.channels
        if out_channel == 0:
            out_channel = channels[self.first_level]
        dbs = ["db10", "db8", "db6", "db4", "db3", "db2"]
        # for i in range(self.first_level)
        self.local_attention = Attribution(channels[-1], 256)
        self.association = CrossUpdate(256)

        self.upOnes = DLAUpOnes(self.first_level, len(channels), channels[self.first_level:])
        self.level_up = LevelSplit(channels[-2], channels[self.first_level], channels[self.first_level],
                                   scale=4, wave=dbs[self.first_level])
        self.upPlus1 = nn.Sequential(
            DeformConv(channels[self.first_level + 1], channels[self.first_level] * 2),
            nn.ConvTranspose2d(channels[self.first_level] * 2, channels[self.first_level] * 2, 4, 2,
                               padding=1, output_padding=0, bias=False, groups=channels[self.first_level])
        )
        self.upPlus2 = nn.Sequential(
            DeformConv(channels[-1], channels[self.first_level] * 2),
            nn.ConvTranspose2d(channels[self.first_level] * 2, channels[self.first_level] * 2, 16, 8,
                               padding=4, groups=out_channel, bias=False, output_padding=0)
        )
        self.out_conv = nn.Sequential(
            nn.InstanceNorm2d(channels[self.first_level]),
            nn.Conv2d(channels[self.first_level], out_channel, 3, 1, 1)
        )
        self.heads = heads
        set_attribute(self, head_conv, channels, final_kernel)

    def forward(self, x, input_type="image", splen=5):
        x = self.base(x)
        x[-1] = self.local_attention(x[-1])
        self.upOnes(x)
        if input_type == "video":
            sp_imgs = x[-2].unflatten(0, (splen, -1))  # splen, bs, flen, h, w
            spimgs_ = []
            self.association.new_sequence()
            for imgs in sp_imgs:
                spimgs_.append(self.association(imgs))

            spimgs_ = torch.cat(spimgs_, dim=0)
            x[-2] = spimgs_
        y = self.level_up([x[self.first_level], x[-2]])
        out = y + self.upPlus1(x[-3]).unflatten(1, (2, -1)).flatten(0, 1) + \
              self.upPlus2(x[-1]).unflatten(1, (2, -1)).flatten(0, 1)

        bkout = self.out_conv(out)
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(bkout)
        return [z]


class ArchSmall(nn.Module):
    def __init__(self, heads, down_ratio, final_kernel, head_conv, out_channel=0):
        super(ArchSmall, self).__init__()
        assert down_ratio in [4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.base = dla34()
        channels = self.base.channels
        if out_channel == 0:
            out_channel = channels[self.first_level]
        dbs = ["db10", "db8", "db6", "db4", "db3", "db2"]
        self.modules_list = nn.ModuleList()
        self.local_attention = Attribution(channels[-1], 256)
        self.association = CrossUpdate(256)

        self.level_up1 = LevelUp(self.first_level, self.first_level + 2,
                                 channels[self.first_level:], dbs[self.first_level:])
        self.level_up2 = LevelUp(len(channels) - 2, len(channels), channels[-2:], dbs[-2:])

        self.dla_up = IDAUp(channels[self.first_level], channels[-3:-1], [2, 4])
        self.out_conv = nn.Sequential(
            nn.InstanceNorm2d(channels[self.first_level]),
            nn.Conv2d(channels[self.first_level], out_channel, 3, 1, 1)
        )
        self.heads = heads
        set_attribute(self, head_conv, channels, final_kernel)

    def forward(self, x, input_type="image", splen=5):
        x = self.base(x)
        x[-1] = self.local_attention(x[-1])
        self.level_up1(x)
        self.level_up2(x)

        if input_type == "video":
            sp_imgs = x[-2].unflatten(0, (splen, -1))  # splen, bs, flen, h, w
            spimgs_ = []
            self.association.new_sequence()
            for imgs in sp_imgs:
                spimgs_.append(self.association(imgs))

            spimgs_ = torch.cat(spimgs_, dim=0)
            x[-2] = spimgs_
        self.dla_up([x[self.first_level], x[-2]], 0, 2)
        bkout = self.out_conv(x[self.first_level])
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(bkout)
        return [z]


def get_pose_net(arch, num_layers, heads, head_conv=256, down_ratio=4, pretrained=False):
    if arch == "large":
        model = ArchLarge(heads, down_ratio, final_kernel=1, head_conv=head_conv, out_channel=0)

    elif arch == "middle":
        model = ArchMiddle(heads, down_ratio, final_kernel=1, head_conv=head_conv, out_channel=0)

    elif arch == "small":
        model = ArchSmall(heads, down_ratio, final_kernel=1, head_conv=head_conv, out_channel=0)

    else:
        model = DLASeg('dla{}'.format(num_layers), heads,
                       pretrained=pretrained,
                       down_ratio=down_ratio,
                       final_kernel=1,
                       last_level=5,
                       head_conv=head_conv)

    return model


def set_attribute(model: nn.Module, head_conv, channels, final_kernel):
    for head in model.heads:
        classes = model.heads[head]
        if head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(channels[model.first_level], head_conv,
                          kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes,
                          kernel_size=final_kernel, stride=1,
                          padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
        else:
            fc = nn.Conv2d(channels[model.first_level], classes,
                           kernel_size=final_kernel, stride=1,
                           padding=final_kernel // 2, bias=True)

            if 'hm' in head:
                fc.bias.data.fill_(-2.19)

            else:
                fill_fc_weights(fc)

        model.__setattr__(head, fc)


if __name__ == '__main__':
    import torch
    from torch.autograd import profiler
    model = get_pose_net("middle", 34, {"box": 4, 'hm': 1, 'reid': 128}, pretrained=False, down_ratio=4,
                     head_conv=128).to("cuda")
    t1 = torch.cuda.max_memory_allocated() / 1024 ** 2.
    imgs = torch.randn((4, 3, 608, 1088)).to("cuda")
    out = model(imgs)[0]
    print({key: v.shape for key, v in out.items()})
    print("memory cost: ", torch.cuda.max_memory_allocated() / 1024 ** 2. - t1)
    # with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
    #     # neighbor_correlation_filter(x, x, radius=1, dilation=2)
    #     out = model2(imgs)
    # print(prof)
