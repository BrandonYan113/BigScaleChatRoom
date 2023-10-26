import cv2
import os
import numpy as np
import glob
from src.lib.trains.train_utils import Annotator


def get_frame_dict(dir, imgs_path=None):
    frame_dict = {}
    imgs = sorted(glob.glob(os.path.join(imgs_path, "*.jpg"))) if imgs_path is not None else None
    with open(dir,'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.replace("\n", "").split(",")
            assert len(line) >= 7, "txt file illegal, len of line < 7"
            frame = int(float(line[0]))
            id = int(float(line[1]))
            left, top = float(line[2]), float(line[3])
            w, h = float(line[4]), float(line[5])
            box = np.array([left, top, left + w, top + h])
            score = float(line[6])
            if score < 0.:
                continue

            if frame not in frame_dict:
                frame_dict[frame] = {"boxes": [box], "img_path": imgs[frame-1] if imgs_path is not None else None,
                                     "ids": [id]}
            else:
                frame_dict[frame]["boxes"].append(box)
                frame_dict[frame]["ids"].append(id)

    return frame_dict


if __name__ == '__main__':
    src_dir = r"D:\A\MOTDataset\MOT17\test"
    det_dir = r"D:\A\MOTDataset\MOT17\results\mot17_20_middle_arch1_with_memory_net_track"
    # src_dir = r"D:\A\MOTDataset\MOT20\test"
    # det_dir = r"D:\A\MOTDataset\MOT20\results\mot17_20_middle_arch1_with_memory_net_track"
    det_dir = r"D:\A\MOTDataset\MOT17\results\mot17_20_middle_arch1_with_memory_net1_id045_no_memory_radius1_track"
    det_dir = r"D:\A\MOTDataset\results\mot17_20_middle_arch1_with_memory_net1_id045_no_memory_radius1_conf04_track"
    # print(seqs)
    seqs_txt = glob.glob(os.path.join(det_dir, "*.txt"))
    seqs = [str(seq).replace("\\", "/").replace(".txt", "").split("/")[-1] for seq in seqs_txt]

    det_or_gt = "det"
    for i, seq in enumerate(seqs):
        src_txt = os.path.join(src_dir, seq, det_or_gt, f"{det_or_gt}.txt")
        src_imgs = os.path.join(src_dir, seq, "img1")
        det_txt = seqs_txt
        # print(src_txt, det_txt)
        src_dict = get_frame_dict(src_txt, src_imgs)
        det_dict = get_frame_dict(det_txt[i], None)

        for key, vales in src_dict.items():
            img = vales["img_path"]
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            img_annt = Annotator(img)
            # for box in vales["boxes"]:
            #     img_annt.box_label(box, color=(0, 255, 0))

            if key in det_dict:
                for box, id in zip(det_dict[key]["boxes"], det_dict[key]["ids"]):
                    img_annt.box_label(box, color=(0, 0, 255), label=str(id), txt_color=(255, 255, 255))

            img = img_annt.result()
            cv2.imshow("img", img)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()