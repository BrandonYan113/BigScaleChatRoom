import os
import glob
import numpy as np


def get_dict(src_dir, sum_keys=None):
    txts = glob.glob(os.path.join(src_dir, "*.txt"))
    seqs = [txt.replace("\\", "/").split("/")[-1] for txt in txts]
    summary = {}
    for seq in seqs:
        data = np.loadtxt(os.path.join(src_dir, seq), delimiter=",", usecols=1, dtype=np.float32)
        summary[seq] = {}
        summary[seq] = {
            "dets_num": len(data),
            "ids": data.max() - data.min() + 1
        }
    total_dets, total_ids = 0, 0
    for key, values in summary.items():
        if sum_keys is None:
            total_dets += values["dets_num"]
            total_ids += values["ids"]
            # print(key, values)

        else:
            if key in sum_keys:
                total_dets += values["dets_num"]
                total_ids += values["ids"]
                # print(key, values)

    # print(total_dets)
    print("total dets: ", total_dets, "total ids: ", total_ids, "ratio: ", float(total_ids) / total_dets)
    return summary


if __name__ == '__main__':
    keys = ["MOT20-04.txt", "MOT20-06.txt", "MOT20-07.txt", "MOT20-08.txt"]
    get_dict(r"D:\BanYanDeng\A\results\results\mot20test", keys)
    get_dict(r"D:\BanYanDeng\MOTDataset\results\mot17_20_test_mot20_conf045_nms075_id05_track", keys)
    get_dict(r"D:\BanYanDeng\MOTDataset\results\mot17_20_test_mot20_conf0425_nms075_id05_track", keys)
    get_dict(r"D:\BanYanDeng\MOTDataset\results\mot17_20_test_mot20_conf040_nms075_id05_buffer50_track", keys)
    get_dict(r"D:\BanYanDeng\MOTDataset\MOT20\results\extra_exp_nms75_conf425_mot20_test_pth_15_id05_track", keys)
    # get_dict(r"D:\BanYanDeng\MOTDataset\results\mot17_20_test_mot20_conf045_nms075_id05_buffer40_with_memory_track", keys)
    # get_dict(r"D:\BanYanDeng\MOTDataset\MOT20\results\extra_exp_nms75_conf45_mot20_test_pth_last_track", keys)
    # get_dict(r"D:\BanYanDeng\MOTDataset\MOT20\results\extra_exp_nms70_conf45_mot20_test_pth_last_track", keys)
    # get_dict(r"D:\BanYanDeng\MOTDataset\MOT20\results\extra_exp_nms70_conf45_mot20_test_pth_15_track", keys)
    # get_dict(r"D:\BanYanDeng\MOTDataset\MOT20\results\extra_exp_nms70_conf425_mot20_test_pth_15_track", keys)
    # get_dict(r"D:\BanYanDeng\MOTDataset\MOT20\results\extra_exp_nms75_conf425_mot20_test_pth_15_track", keys)
    # get_dict(r"D:\BanYanDeng\MOTDataset\MOT20\results\extra_exp_nms75_conf425_mot20_test_pth_last_id05_track", keys)
    # get_dict(r"D:\BanYanDeng\MOTDataset\MOT20\results\extra_exp_nms75_conf45_mot20_test_5_last_id05_track", keys)
    # get_dict(r"D:\BanYanDeng\MOTDataset\MOT20\results\extra_exp_nms75_conf45_mot20_test_pth_last_id05_track", keys)
    print("=" * 40, "mot17 half val compare")
    # get_dict(r"")
    keys = ["MOT17-01-SDP.txt", "MOT17-03-SDP.txt", "MOT17-06-SDP.txt", "MOT17-07-SDP.txt", "MOT17-08-SDP.txt",
            "MOT17-12-SDP.txt", "MOT17-14-SDP.txt"]

    get_dict(r"D:\BanYanDeng\A\results\results\mot17test", keys)
    get_dict(r"D:\BanYanDeng\MOTDataset\results\mot17_20_extra_exp_nms75_conf45_track", keys)
    get_dict(r"D:\BanYanDeng\MOTDataset\MOT17\results\extra_exp_nms70_conf45_mot17_test_pth_last_id055_track", keys)
    get_dict(r"D:\BanYanDeng\MOTDataset\MOT17\results\extra_exp_nms70_conf45_mot17_test_pth_15_id055_track", keys)

