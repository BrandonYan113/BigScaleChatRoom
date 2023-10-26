import os
import glob


def divide_train_set(img_dir, target_dir, seq, prefix):
    half = os.path.join(target_dir, "mot20.half")
    val = os.path.join(target_dir, "mot20.val")
    hf = open(half, 'a')
    vf = open(val, 'a')
    hlines, vlines = [], []
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    for i, img_path in enumerate(imgs):
        img_name = os.path.basename(img_path)
        line = os.path.join(prefix.replace("REPLACE_", seq), img_name) + "\n"
        if i < int(len(imgs) / 2):
            hlines.append(line)
        else:
            vlines.append(line)

    hf.writelines(hlines)
    vf.writelines(vlines)
    hf.close()
    vf.close()


if __name__ == '__main__':
    target_dir = "../src/data"
    original_dir = r"D:\BanYanDeng\MOTDataset\MOT20\train"
    prefix = "MOT20/images/train/REPLACE_/img1"
    seqs = sorted(os.listdir(original_dir))

    for seq in seqs:
        img_dir = os.path.join(original_dir, seq, "img1")
        divide_train_set(img_dir, target_dir, seq, prefix)
