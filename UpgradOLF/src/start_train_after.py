import train
from src.lib.opts import opts
import time
import os


if __name__ == '__main__':
    opt = opts().parse()
    opt.task = "mot"
    opt.exp_id = "danceTrack_all_arch1_with_memory"
    opt.arch_scale = "middle_"
    opt.gpus = [0]
    opt.pretrained = True
    opt.load_model = "../exp/mot/crowdhuman_middle_arch1/model_last.pth"
    opt.num_epochs = 20
    opt.lr_step = [15]
    opt.data_cfg = "../src/lib/cfg/danceTrackALl.json"
    opt.level_num = 2
    opt.conf_thres = 0.3
    opt.id_weight = 0.4
    opt.wh_weight = 0.5
    opt.input_type = 'image'
    opt.lr = 1e-4
    opt.batch_size = 10
    opt.print_or_show_iter = 100
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    wait_time = 15 * 14 * 60
    # wait_time = 2
    for i in range(1, wait_time):
        print("\r run train.py after {:d}s".format((wait_time - i)), end="")
        time.sleep(1)
    train.main(opt)
