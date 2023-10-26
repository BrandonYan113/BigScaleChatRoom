import os, os.path as osp
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def gen_train_txt(seq_root, label_root, train_txt_root, prefix):
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]
    train_txt = open(train_txt_root, "w")
    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        if seq.startswith("dance"):
            seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

            gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            seq_label_root = osp.join(label_root, seq, 'img1')
            mkdirs(seq_label_root)
            # print(seq_root, seq, gt.shape)
            for fid, tid, x, y, w, h, mark, label, _ in gt:
                if mark == 0 or not label == 1:
                    continue

                fid = int(fid)
                tid = int(tid)
                if not tid == tid_last:
                    tid_curr += 1
                    tid_last = tid
                x += w / 2
                y += h / 2
                label_fpath = osp.join(seq_label_root, '{:08d}.txt'.format(fid))
                label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
                with open(label_fpath, 'a') as f:
                    f.write(label_str)

            imgs_root = os.listdir(osp.join(seq_root, seq, "img1"))
            for img in imgs_root:
                train_txt.write(osp.join(prefix, seq, "img1", img)+"\n")

    train_txt.close()
    print("Done!")


if __name__ == '__main__':
    seq_root = r'D:\A\DanceTrack\train1'
    label_root = '/data/DanceTrack/labels_with_ids/train1'
    train_txt_root = "../src/data/danceTrackTrain1.train"
    prefix = r"DanceTrack\train1"
    gen_train_txt(seq_root, label_root, train_txt_root, prefix)

    seq_root = r"D:\A\DanceTrack\train2"
    label_root = '/data/DanceTrack/labels_with_ids/train2'
    train_txt_root = "../src/data/danceTrackTrain2.train"
    prefix = r"DanceTrack\train2"

    gen_train_txt(seq_root, label_root, train_txt_root, prefix)
