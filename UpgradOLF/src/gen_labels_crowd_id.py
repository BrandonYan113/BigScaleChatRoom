import os.path as osp
import os
import cv2
import json
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records


def gen_labels_crowd(data_root, label_root, ann_root, train=False, limit_id_nums=0, gen_train_txt=None):
    mkdirs(label_root)
    anns_data = load_func(ann_root)
    if gen_train_txt is not None:
        train_txt = open(gen_train_txt, 'w')
    tid_curr = 0
    for i, ann_data in enumerate(anns_data):
        image_name = '{}.jpg'.format(ann_data['ID'])
        flag = False
        if train:
            for i in range(3):
                img_path = os.path.join(data_root, f"CrowdHuman_train0{i+1}/Images", image_name)
                if os.path.exists(img_path):
                    flag = True
                    if gen_train_txt is not None:
                        train_txt.write(osp.join("crowdhuman/images/train", image_name) + "\n")
                    break
            if not flag:
                msg = f"not found img_pth: {img_path}"
                raise ValueError(msg)
        else:
            img_path = os.path.join(data_root, image_name)
        # print(img_path)
        anns = ann_data['gtboxes']
        img = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img_height, img_width = img.shape[0:2]
        for i in range(len(anns)):
            if 'extra' in anns[i] and 'ignore' in anns[i]['extra'] and anns[i]['extra']['ignore'] == 1:
                continue
            x, y, w, h = anns[i]['fbox']
            x += w / 2
            y += h / 2
            label_fpath = os.path.join(label_root, os.path.basename(img_path.replace("jpg", 'txt')))
            # label_fpath = img_path.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / img_width, y / img_height, w / img_width, h / img_height)

            with open(label_fpath, 'a') as f:
                f.write(label_str)
            tid_curr += 1

        if 0 < limit_id_nums < tid_curr:
            break
    if gen_train_txt is not None:
        train_txt.close()


if __name__ == '__main__':
    data_val = 'D:\A\CrowdHuman\CrowdHuman_val\Images'
    label_val = 'D:/data/crowdhuman/labels_with_ids/val'
    ann_val = 'D:\A\CrowdHuman/annotation_val.odgt'

    label_train = 'D:/data/crowdhuman/labels_with_ids/train'
    ann_train = 'D:\A\CrowdHuman/annotation_train.odgt'
    data_train = "D:\A\CrowdHuman"
    lmit_num = 2000
    label_train_limit = f'D:/data/crowdhuman_{lmit_num}/labels_with_ids/train'
    ann_train_limit = 'D:\A\CrowdHuman/annotation_train.odgt'
    data_train_limit = "D:\A\CrowdHuman"

    # gen_labels_crowd(data_train, label_train, ann_train, train=True)
    # gen_labels_crowd(data_val, label_val, ann_val, train=False)
    gen_labels_crowd(data_train_limit, label_train_limit, ann_train_limit, train=True, limit_id_nums=lmit_num,
                     gen_train_txt=f"data/crowdhuman_{lmit_num}.train")


