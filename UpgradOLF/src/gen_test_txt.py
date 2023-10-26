import os


def gen_mot_test_txt(orig_root, target_root, txt_name, format, set=None):
    with open(os.path.join(target_root, txt_name), mode='w', encoding='utf-8') as txt:
        seqs = [s for s in os.listdir(orig_root)]
        for seq in seqs:
            imgs_base = None
            if set is not None:
                if set in seq:
                    imgs_base = os.path.join(orig_root, seq, "img1")
            else:
                imgs_base = os.path.join(orig_root, seq, "img1")

            if imgs_base is None:
                continue

            imgs = os.listdir(imgs_base)
            for img in imgs:
                img_path = os.path.join(format, seq, "img1", img)
                txt.write(img_path + '\n')


if __name__ == '__main__':
    # root = "D:\BanYanDeng\MOTDataset\MOT17\\test"
    # target_root = "data"
    # txt_name = "mot17.test"
    # gen_mot_test_txt(root, target_root, txt_name, "MOT17/images/test", "SDP")

    root = "D:\BanYanDeng\MOTDataset\MOT20\\test"
    target_root = "data"
    txt_name = "mot20.test"
    gen_mot_test_txt(root, target_root, txt_name, "MOT20/images/test")