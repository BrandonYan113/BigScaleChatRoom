import os.path as osp
import os
import glob
import shutil
import numpy as np


def gen_val(seq):
    seq_name = osp.basename(seq)
    val = osp.join(osp.dirname(osp.dirname(seq)), "val")
    val = osp.join(val, seq_name)
    if not osp.exists(val):
        os.makedirs(val)

    dirs = os.listdir(seq)
    for dir in dirs:
        if dir.endswith(".ini"):
            scr_ini = osp.join(seq, dir)
            tar_ini = osp.join(val, dir)
            shutil.copy(scr_ini, tar_ini)
        if dir == "img1":
            imgs = sorted(glob.glob(osp.join(seq, dir, "*.jpg")))

            if not osp.exists(osp.join(val, dir)):
                os.mkdir(osp.join(val, dir))

            for i, img in enumerate(imgs):
                if i >= int(len(imgs) / 2):
                    img_name = osp.basename(img)
                    tar_img = osp.join(val, dir, img_name)
                    shutil.copy(img, tar_img)

        if dir == 'gt':
            gt = osp.join(seq, dir, "gt.txt")
            tar_gt = osp.join(val, dir, "gt.txt")
            if not osp.exists(osp.join(val, dir)):
                os.mkdir(osp.join(val, dir))
            tar_gt_lines = []
            with open(gt, 'r') as file:
                lines = file.readlines()
                lines = np.array([line.replace("\n", "").split(",") for line in lines], dtype=np.float32)
                frame_nums = np.max(lines[:, 0])
                for line in lines:
                    frame_id = int(line[0])
                    half_ = int(frame_nums / 2)
                    if frame_id > half_:
                        line = [str(int(line[0]) - half_)] + [str(elm) for elm in line[1:]]
                        suffix = ",-1" * (10 - len(line))
                        tar_line = ",".join(line) + suffix + "\n"

                        tar_gt_lines.append(tar_line)

            val_gt_file = open(tar_gt, 'w')
            val_gt_file.writelines(tar_gt_lines)
            val_gt_file.close()


if __name__ == '__main__':
    dataset = r"D:\A\MOTDataset\MOT17SDP\train"
    for seq in os.listdir(dataset):
        gen_val(osp.join(dataset, seq))

    dataset = r"D:\A\MOTDataset\MOT20\train"
    for seq in os.listdir(dataset):
        gen_val(osp.join(dataset, seq))
