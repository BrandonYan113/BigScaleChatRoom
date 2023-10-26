import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def gen_train_txt(seq_root, label_root):
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        seq_label_root = osp.join(label_root, seq, 'img1')
        mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, vis in gt:
            if not (mark == 1 and label == 1 and vis > 0.1):
                continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)


def gen_test_txt(seq_root, label_root):
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        gt_txt = osp.join(seq_root, seq, 'det', 'det.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        seq_label_root = osp.join(label_root, seq, 'img1')
        mkdirs(seq_label_root)
        gt = gt[:, :7]
        for fid, tid, x, y, w, h, mark in gt:
            if mark == 0:
                continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)


if __name__ == '__main__':
    seq_root = r'D:\BanYanDeng\MOTDataset\MOT17\train'
    label_root = '/data/MOT17/labels_with_ids/train'
    gen_train_txt(seq_root, label_root)

    # seq_root = r"D:\A\MOTDataset\MOT17\test"
    # label_root = '/data/MOT17/labels_with_ids/test'
    #
    # gen_test_txt(seq_root, label_root)
