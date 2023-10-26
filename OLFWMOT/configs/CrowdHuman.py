import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from support.genenralized_rcnn_transform import GeneralizedRCNNTransform
from utils.annotation import Annotator


class CrowdHuman(Dataset):
    def __init__(self, datapth, pre_transform=None, train=True, shuffle=True):
        super(CrowdHuman, self).__init__()
        self.train = train
        self.datapth = datapth
        self.pre_transform = pre_transform
        self.trainDataset = ["CrowdHuman_train01\Images", "CrowdHuman_train02\Images",
                         "CrowdHuman_train03\Images"]
        self.test_dataset = ["CrowdHuman_val\Images", "CrowdHuman_test\Images"]
        if train:
            odgt_pth = os.path.join(datapth, "annotation_train.odgt")
        else:
            odgt_pth = os.path.join(datapth, "annotation_val.odgt")

        self.annotator = self.load_odgt(odgt_pth)
        if shuffle:
            np.random.shuffle(self.annotator)
        self.len = len(self.annotator)

    def __getitem__(self, index):
        gt = self.annotator[index]
        name = gt['ID'] + '.jpg'
        targets = {'boxes': [], 'labels': []}
        for box in gt['gtboxes']:
            label = box['tag']
            if label == 'person':
                targets['labels'].append(1)
                box = box['vbox']
                box[2] += box[0]
                box[3] += box[1]
                targets['boxes'].append(box)

        targets['boxes'] = torch.tensor(targets['boxes'])
        targets['labels'] = torch.tensor(targets['labels'])

        for dataset in self.trainDataset:
            img_path = os.path.join(self.datapth, dataset, name)
            if os.path.exists(img_path):

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.pre_transform is not None:
                    img, target = self.pre_transform([img], [targets])
                if len(img) > 0:
                    return img[0], target[0]
                else:
                    return None, None

    def __len__(self):
        return self.len

    def load_odgt(self, odgt_pth):  # str to list
        assert os.path.exists(odgt_pth)  # assert() raise-if-not
        with open(odgt_pth, 'r') as fid:
            lines = fid.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]  # str to list

        return records


class CrowdHumanMiniBatch(Dataset):
    def __init__(self,
                 datapth,
                 transform: GeneralizedRCNNTransform,
                 level_nums=4,
                 scale_factor=8,
                 pre_transform=None,
                 train=True,
                 batch_size=4,
                 shuffle=True,
                 box_label_func='default',
                 min_area=16. * 16.,
                 label_min_overlap_ratio=0.5
                 ):
        super(CrowdHumanMiniBatch, self).__init__()
        self.dataset = CrowdHuman(datapth, pre_transform, train, shuffle=shuffle)
        self.len = len(self.dataset) // batch_size + len(self.dataset) % batch_size
        self.batch_size = batch_size
        self.start_index = 0
        self.transform = transform
        self.box_label_func = box_label_func
        self.scale_factor = scale_factor
        self.level_nums = level_nums
        self.min_area = min_area

        self.Annotator = Annotator(scale_factor, level_nums, min_area,
                                   label_min_overlap_ratio=label_min_overlap_ratio)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        imgs, targets = [], []
        for i in range(self.batch_size):
            img, target = self.dataset[i + self.start_index]
            if img is not None:
                imgs.append(img)
                targets.append(target)

        self.start_index += len(imgs)

        imgs, targets = self.transform(imgs, targets)
        bboxes = [target['boxes'] for target in targets]
        imgs = imgs.tensors
        h, w = imgs.shape[-2:]
        batch_shape = (int(h / self.scale_factor), int(w / self.scale_factor))
        boxes, heat_maps = self.Annotator.current_frame(bboxes,
                                                        batch_shape,
                                                        box_label_func=self.box_label_func,
                                                        )
        # boxes_, _ = self.Annotator.current_frame(bboxes, batch_shape,
        #                                          box_label_func="patch")

        return imgs, boxes, heat_maps, 'image'
