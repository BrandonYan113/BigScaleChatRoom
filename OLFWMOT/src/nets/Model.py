import _init_paths
import torch
from torch import nn
from nets.backbone import Backbone
from nets.subnet import DetectHead, AdaptiveTemplate, LevelSplit, Attribution, Bridge
from nets.memory import MemoryNet


class StrongModel(nn.Module):
    def __init__(self,
                 backbone_name='default',
                 level_num=2,
                 box_num=3,
                 split_len=4,
                 level_channel=256,
                 memory_channel=128,
                 subnet_mode="DETECT_ONLY",
                 save_levels=False,
                 bridge_kernel=[3, 1, 1, 3]
                 ):
        '''
        :param adaptiveMode: ['adaptive', 'default']
        '''

        super(StrongModel, self).__init__()
        self.split_channel = level_channel
        self.backbone = Backbone(backbone_name)
        self.level_split = LevelSplit(512, 256, level_channel, level_num)
        self.attribution_head = Attribution(level_channel, save_levels=save_levels)
        self.adaptive_head = AdaptiveTemplate(level_channel)
        if subnet_mode not in ['DETECT_ONLY', "WITH_LAYERING", 'WITH_ATTRIBUTION', 'WITH_ASSOCIATION']:
            msg = f"undefined subnet mode {subnet_mode}, it should be in ['DETECT_ONLY', " \
                  f"'WITH_LAYERING', 'WITH_ATTRIBUTION', 'WITH_ASSOCIATION']"
            raise ValueError(msg)

        if subnet_mode == 'DETECT_ONLY':
            self.memory_head = MemoryNet(512, feature_len=memory_channel)
        else:
            self.memory_head = MemoryNet(256, feature_len=memory_channel)

        self.bridge_head = Bridge(level_channel, bridge_kernel)
        self.detect_head = DetectHead(level_channel, box_num=box_num)
        self.subnet_mode = subnet_mode
        self.level_num = level_num
        self.box_num = box_num
        self.split_len = split_len
        self.cur_id_feature = None

        self.init_weight()

    def default_mode(self, imgs, loc1=None, loc2=None, input_type='image'):
        '''
        :param imgs: (bs, 3, h, w) if input_type=='image' else (split * bs, 3, h, w)
        :param input_type: ['image', 'video']
        :return:
        '''
        if input_type == 'image':
            x = self.backbone(imgs)
            x = self.level_split(x)   # n * level_num, level_channel, h, w

            if self.subnet_mode in ["WITH_ATTRIBUTION", "WITH_ASSOCIATION"]:
                x = self.attribution_head(x)

            bf = self.bridge_head(x)
            boxes, heat_maps = self.detect_head(bf)  # n * level_num, level_channel, h, w

            # n * level_num * box_num, 4, h, w
            boxes = boxes.unflatten(1, (self.box_num, 4)).flatten(0, 1)

            # n * level_num * box_num, 1, h, w
            heat_maps = heat_maps.unflatten(1, (self.box_num, 1)).flatten(0, 1)
            return boxes, heat_maps

        elif input_type == 'video':
            l3, l4 = self.backbone(imgs)    # splen * bs, c, h, w
            l3 = l3.unflatten(0, (self.split_len, -1))   # split, bs, c3, h, w
            l4 = l4.unflatten(0, (self.split_len, -1))   # split, bs, c4, h / 2, w / 2

            split_boxes, split_heat_maps = [], []
            split_corr, memory_corr = [], []
            memory_corr_nums1, memory_corr_nums2 = [], []
            self.adaptive_head.new_sequence()   # initial template
            for i, f in enumerate(zip(l3, l4)):  # split enumerate
                f = self.level_split(f)     # bs * level_num, level_channel, h, w

                if not self.subnet_mode == "WITH_LAYERING":
                    f = self.attribution_head(f)

                if self.subnet_mode == 'WITH_ASSOCIATION':
                    f = self.adaptive_head(f, input_type)  # bs * level_num, level_channel, h, w
                    lastf = self.adaptive_head.last_template
                    curf = self.adaptive_head.template

                f = self.bridge_head(f)
                if self.subnet_mode in ["WITH_LAYERING", "WITH_ATTRIBUTION"]:
                    if i > 0:
                        lastf = curf
                    curf = f

                boxes, heat_maps = self.detect_head(f)
                # bs * level_num * box_num, 4, h, w
                boxes = boxes.unflatten(1, (self.box_num, 4)).flatten(0, 1)
                # bs * level_num * box_num, 1, h, w
                heat_maps = heat_maps.unflatten(1, (self.box_num, 1)).flatten(0, 1)

                split_boxes.append(boxes)
                split_heat_maps.append(heat_maps)

                if i > 0 and loc1 is not None and loc2 is not None:
                    # bs, level_nums, level_nums, neighbor_num, h, w
                    lf = lastf.unflatten(0, (-1, self.level_num))
                    cf = curf.unflatten(0, (-1, self.level_num))

                    cur_f = cf
                    last_f = lf

                    locations1, locations2 = loc1[i-1], loc2[i-1]
                    mcorr, nums1, nums2 = self.memory_association(locations1, locations2, last_f, cur_f)
                    memory_corr.append(mcorr)
                    memory_corr_nums1.append(nums1)
                    memory_corr_nums2.append(nums2)

            # splen * bs * level_num * box_num, 4, h, w
            split_boxes = torch.cat(split_boxes, dim=0)
            split_heat_maps = torch.cat(split_heat_maps, dim=0)

            return split_boxes, split_heat_maps, memory_corr, memory_corr_nums1, memory_corr_nums2
        else:
            raise ValueError("input_type should be image or video")

    def detect_only(self, x, loc1=None, loc2=None, input_type='image'):
        _, l4 = self.backbone(x)
        l4 = self.bridge_head(l4)
        boxes, heat_maps = self.detect_head(l4)
        boxes = boxes.unflatten(1, (self.box_num, 4)).flatten(0, 1)
        heat_maps = heat_maps.unflatten(1, (self.box_num, 1)).flatten(0, 1)

        memory_corr_nums1, memory_corr_nums2, memory_corr = [], [], []
        if input_type == 'video':
            l4 = l4.unflatten(0, (self.split_len, -1, self.level_num))  # split, bs, level_num, c4, h / 2, w / 2
            for i in range(len(l4)):
                if i > 0 and loc1 is not None and loc2 is not None:
                    # 1, level_nums, level_nums, neighbor_num, h, w
                    locations1, locations2 = loc1[i-1], loc2[i-1]
                    l_0, l_1 = l4[i-1], l4[i]
                    f1, nums1 = self.memory_head.sample_feature(l_0, locations1)  # n, sum(mi), feature_len, 1
                    f2, nums2 = self.memory_head.sample_feature(l_1, locations2)  # n, sum(mi), feature_len, 1
                    if not f1 == [] or not f2 == []:
                        corr = self.memory_head(f1, f2)  # n, m, m
                    memory_corr.append(corr)
                    memory_corr_nums1.append(nums1)
                    memory_corr_nums2.append(nums2)

        return boxes, heat_maps, memory_corr, memory_corr_nums1, memory_corr_nums2

    def test_mode(self, img, input_type='image'):
        # img: 1, 3, h, w
        l3_4 = self.backbone(img)  # bs, channel, h / 2, w / 2
        if self.subnet_mode == "DETECT_ONLY":
            x = l3_4[1]
        elif self.subnet_mode in ["WITH_LAYERING", "WITH_ATTRIBUTION", "WITH_ASSOCIATION"]:
            x = self.level_split(l3_4)  # n * level_num, level_channel, h, w
            if not self.subnet_mode == "WITH_LAYERING":
                x = self.attribution_head(x)

            if self.subnet_mode == 'WITH_ASSOCIATION' and input_type == 'video':
                x = self.adaptive_head(x)   # n * level_num, level_channel, h, w
        else:
            msg = f"undefined subnet mode {self.subnet_mode}"
            raise ValueError(msg)

        x = self.bridge_head(x)
        if self.subnet_mode == "DETECT_ONLY":
            self.cur_id_feature = x
        elif self.subnet_mode == "WITH_ASSOCIATION":
            template = self.adaptive_head.template
            # self.cur_id_feature = torch.cat((dec, template), dim=1)
            self.cur_id_feature = template
        else:
            # self.cur_id_feature = torch.cat((dec, x), dim=1)
            self.cur_id_feature = x

        boxes, heat_maps = self.detect_head(x)

        # level_num * box_num, 4, h, w
        boxes = boxes.unflatten(1, (self.box_num, 4)).flatten(0, 1)

        # level_num * box_num, 1, h, w
        heat_maps = heat_maps.unflatten(1, (self.box_num, 1)).flatten(0, 1)
        return boxes, heat_maps

    def forward(self, imgs, loc1=None, loc2=None, input_type='image'):
        if self.training:
            if self.subnet_mode == 'DETECT_ONLY':
                return self.detect_only(imgs, loc1,  loc2, input_type=input_type)
            else:
                return self.default_mode(imgs, loc1, loc2, input_type=input_type)
        else:
            return self.test_mode(imgs, input_type)

    def memory_association(self, locations1, locations2, last_f, cur_f):
        f1, nums1 = self.memory_head.sample_feature(last_f, locations1)    # n, sum(mi), feature_len, 1
        f2, nums2 = self.memory_head.sample_feature(cur_f, locations2)    # n, sum(mi), feature_len, 1
        corr = None
        if not f1 == [] or not f2 == []:
            corr = self.memory_head(f1, f2)     # n, m, m

        return corr, nums1, nums2

    def new_sequence(self):
        self.adaptive_head.new_sequence()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)


def get_model(config):
    model = StrongModel(config.TRAINING.BACKBONE_NAME,
                        config.TRAINING.LEVEL_NUM,
                        box_num=config.TRAINING.BOX_NUM,
                        split_len=config.TRAINING.SPLIT_LEN,
                        level_channel=config.TRAINING.LEVEL_CHANNEL,
                        subnet_mode=config.TRAINING.SUBNET_MODE,
                        save_levels=config.TRAINING.SAVE_LEVELS,
                        bridge_kernel=config.TRAINING.BRIDGE_KERNEL
                        )

    return model
