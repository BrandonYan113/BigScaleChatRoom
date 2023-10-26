import torch
from src.lib.datasets.dataset.jde import JointDataset
from src.lib.opts import opts
import json
import os
import os.path as osp
import importlib.resources as res
from src import data as data
import src
import build
import sys
from src import resource
# import data
import cv2
from src.lib.trains.train_utils import Annotator


if __name__ == '__main__':
    opt = opts().parse()
    data_cfg = osp.join(osp.dirname(osp.dirname(__file__)), "cfg/crowdhuman.json")
    # data_cfg = osp.join(osp.dirname(osp.dirname(__file__)), "cfg/mot17.json")
    f = open(data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    dataset = JointDataset(opt, dataset_root, trainset_paths, augment=True)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    for i, data in enumerate(train_loader):
        if i == 0:
            print(data.keys())  # dict_keys(['input', 'hm', 'reg_mask', 'ind', 'wh', 'reg', 'ids', 'bbox'])

        for img in data["input"]:
            img = img.permute(1, 2, 0).detach().numpy()
            # print(img.shape)
            cv2.imshow("img", img)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWidows()