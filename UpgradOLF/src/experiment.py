import subprocess


def run_cmd(cmd):
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(process.stdout)
    print(process.stderr)


if __name__ == '__main__':
    cmd = ("python track.py --exp_id crowdhuman_val_mot17_middle_arch1 --data_dir D:\A\MOTDataset --pretrained True "
           "--load_model ..\exp\mot\crowdhuman_middle_arch1/model_last.pth --val_mot17 True --arch_scale middle_ "
           "--conf_thres 0.4 --nms_thres 0.7 --input_type video --show_image True --challenge_name mot17_half-val")

    cmd = ("python track.py --exp_id crowdhuman_val_mot20_middle_arch1 --data_dir D:\A\MOTDataset --pretrained True "
           "--load_model ..\exp\mot\crowdhuman_middle_arch1/model_last.pth --val_mot20 True --arch_scale middle_ "
           "--conf_thres 0.4 --nms_thres 0.7 --input_type video --show_image True --challenge_name mot20_half-val")

    cmd = ("python track.py --exp_id mot17_half_val_mot17_middle_arch1 --data_dir D:\A\MOTDataset --pretrained True "
           "--load_model ..\exp\mot\mot17_middle_arch1/model_20.pth --val_mot17 True --arch_scale middle_ "
           "--track_buffer 30 "
           "--conf_thres 0.45 --nms_thres 0.7 --show_image True --challenge_name mot17_half-val")

    cmd = ("python track.py --exp_id mot20_half_val_mot20_middle_arch1 --data_dir D:\A\MOTDataset --pretrained True "
           "--load_model ..\exp\mot\mot20_middle_arch1/model_last.pth --val_mot20 True "
           "--arch_scale middle_ --track_buffer 30 --conf_thres 0.45 --nms_thres 0.7 "
           "--show_image True --challenge_name mot20_half-val")

    cmd = ("python track.py --exp_id danceTrackTrain1_middle_arch1 --data_dir D:\A\DanceTrack --pretrained True "
           "--load_model ..\exp\mot\danceTrackTrain1_middle_arch1/model_last.pth --val_danceTrack True "
           "--arch_scale middle_ --track_buffer 120 --conf_thres 0.4 --nms_thres 0.7 "
           "--show_image True --challenge_name danceTrack-val")

    cmd = (" python track.py --exp_id mot17_half_middle_arch1_with_memory_net --data_dir "
           "D:\A\MOTDataset --pretrained True --load_model ..\exp\mot\mot17_middle_arch1_with_memory_net/model_last.pth "
           "--val_mot17 True --arch_scale middle_ --track_buffer 30 --conf_thres 0.48 --nms_thres 0.7 "
           "--show_image True --challenge_name mot17_half-val ")

    cmd = (" python track.py --exp_id mot20_half_middle_arch1_with_memory_net --data_dir "
           "D:\A\MOTDataset --pretrained True --load_model ..\exp\mot\mot20_middle_arch1_with_memory_net/model_last.pth "
           "--val_mot20 True --arch_scale middle_ --track_buffer 40 --conf_thres 0.45 --nms_thres 0.7 "
           "--show_image True --challenge_name mot20_half-val ")

    cmd = ("python track.py --exp_id mot17_20_middle_arch1_with_memory_net --data_dir D:\A\MOTDataset --pretrained True "
           "--load_model ..\exp\mot\mot17_20_middle_arch1_with_memory_net/model_25.pth --test_mot17 True "
           "--arch_scale middle_ --track_buffer 40 --conf_thres 0.45 --nms_thres 0.7 --show_image True "
           "--challenge_name mot17-test --track_eval_dir None")

    cmd = (
        "python track.py --exp_id mot17_20_middle_arch1_with_memory_net --data_dir D:\A\MOTDataset --pretrained True "
        "--load_model ..\exp\mot\mot17_20_middle_arch1_with_memory_net/model_25.pth --test_mot20 True "
        "--arch_scale middle_ --track_buffer 40 --conf_thres 0.45 --nms_thres 0.7 --show_image True "
        "--challenge_name mot20-test --track_eval_dir None")

    cmd = (
        " python track.py --exp_id danceTrack_middle_arch1_with_memory_net --data_dir D:\A\DanceTrack "
        "--pretrained True --load_model ..\exp\mot\danceTrack_All_middle_arch1_with_memory_net/model_last.pth "
        "--val_danceTrack True --arch_scale middle_ --track_buffer 120 --conf_thres 0.35 --nms_thres 0.7 "
        "--show_image True --challenge_name danceTrack-val --kalman_lambda 0.01 --id_match_thres 0.5"
    )

    # cmd = ("python track.py --exp_id ch_mot17_half_middle_arch_test --data_dir D:\A\MOTDataset --pretrained True "
    #        "--load_model ..\exp\mot\ch_mot17_half_middle_arch_video/model_last.pth --val_mot17 True "
    #        "--conf_thres 0.45 --nms_thres 0.5 --input_type image --show_image True --challenge_name mot17_half-val")
    #
    # cmd = ("python track.py --exp_id mot17_half_middle_arch_image --data_dir D:\A\MOTDataset --pretrained True "
    #        "--load_model ..\exp\mot\mot17_half_middle_arch_image/model_last.pth --val_mot17 True "
    #        "--conf_thres 0.45 --nms_thres 0.5 --input_type image --show_image True --challenge_name mot17_half_image-val")
    #
    # cmd = ("python track.py --exp_id crowdhuman_middle_arch_val_mot17 --data_dir D:\A\MOTDataset --pretrained True "
    #        "--arch_scale middle_ --gpus 0 "
    #        "--load_model ..\exp/mot/crowdhuman_middle_arch1/model_last.pth --val_mot17 True --conf_thres 0.35 "
    #        "--nms_thres 0.5 --input_type image --show_image True --challenge_name crowdhuman_val_mot17-val")
    #
    # cmd = ("python track.py --exp_id mot17_half_middle_arch1_image_val_mot17_image --arch_scale middle_ --gpus 0 "
    #        "--data_dir D:\A\MOTDataset --pretrained True --load_model ..\exp/mot/mot17_half_middle_arch1/model_last.pth "
    #        "--val_mot17 True --conf_thres 0.45 "
    #        "--nms_thres 0.7 --input_type image --show_image True --challenge_name mot17_half_val_mot17_image-val")


    # cmd = (
    #     "python track.py --exp_id mot17_half__middle_arch1_image_val_mot17_image --arch_scale middle_ --gpus 0 "
    #     "--data_dir D:\A\MOTDataset --pretrained True --load_model "
    #     "..\exp/mot/mot17_half_middle_arch1_image/model_30.pth "
    #     "--val_mot17 True --conf_thres 0.4 "
    #     "--nms_thres 0.7 --input_type image --show_image True --challenge_name mot17_half_val_mot17_image-val")

    # cmd = ("python track.py --exp_id mot17_half_mass_order_middle_arch_val_mot17_video --arch_scale middle_ --gpus 0 "
    #        "--data_dir D:\A\MOTDataset --pretrained True --load_model "
    #        "..\exp/mot/mot17_half_mass_order_middle_arch1_video/model_last.pth "
    #        "--val_mot17 True --conf_thres 0.4 --flush_frq 10 "
    #        "--nms_thres 0.5 --input_type video --show_image True "
    #        "--challenge_name mot17_half_mass_order_val_mot17_video-val")
    #
    # cmd = ("python track.py --exp_id crowdhuman_middle_arch_val_mot20 --arch_scale middle_ --gpus 0 "
    #        "--data_dir D:\A\MOTDataset --pretrained True --load_model "
    #        "..\exp/mot/crowdhuman_middle_arch1/model_last.pth "
    #        "--val_mot20 True --conf_thres 0.4 --flush_frq 10 "
    #        "--nms_thres 0.5 --input_type image --show_image True "
    #        "--challenge_name crowdhuman_val_mot20-val")
    #
    # cmd = ("python track.py --exp_id mot20_half_middle_arch_val_mot20_image --arch_scale middle_ --gpus 0 "
    #        "--data_dir D:\A\MOTDataset --pretrained True --load_model "
    #        "..\exp/mot/mot20_half_middle_arch1_image/model_last.pth "
    #        "--val_mot20 True --conf_thres 0.45 --flush_frq 10 "
    #        "--nms_thres 0.7 --input_type image --show_image True "
    #        "--challenge_name mot20_half_val_mot20_image-val")

    # cmd = ("python track.py --exp_id mot20_half_middle_arch1_image_val_mot20_image --arch_scale middle_ --gpus 0 "
    #        "--data_dir D:\A\MOTDataset --pretrained True --load_model "
    #        "..\exp\mot\mot20_half_middle_arch1_image/model_last.pth "
    #        "--val_mot20 True --conf_thres 0.4 "
    #        "--nms_thres 0.7 --input_type image --show_image True "
    #        "--challenge_name mot20_half_image_val_mot20-val")

    # cmd = ("python track.py --exp_id danceTrackTrain1_middle_arch_val --arch_scale middle_ --gpus 0 "
    #        "--data_dir D:\A\DanceTrack --pretrained True --load_model "
    #        "..\exp/mot/danceTrackTrain1_middle_arch1_image/model_last.pth "
    #        "--val_mot20 True --conf_thres 0.45 --flush_frq 10 "
    #        "--nms_thres 0.8 --input_type image --show_image True "
    #        "--challenge_name danceTrackTrain1-val")
    cmd = ("python track.py --exp_id mot17_20_middle_arch1_with_memory_net1_id045_no_memory_radius1 "
           "--data_dir D:\A\MOTDataset --pretrained True --load_model "
           "..\exp\mot\mot17_20_middle_arch1_with_memory_net1/model_last.pth --test_mot20 True "
           "--arch_scale middle_ --track_buffer 40 --conf_thres 0.45 --nms_thres 0.7 --show_image True "
           "--challenge_name mot20-test --track_eval_dir None --id_match_thres 0.45")

    cmd  = ("python track.py --exp_id mot17_20_middle_arch1_with_memory_net1_id045_no_memory_radius1 "
           "--data_dir D:\A\MOTDataset --pretrained True --load_model "
           "..\exp\mot\mot17_20_middle_arch1_with_memory_net1/model_last.pth --test_mot20_all True "
           "--arch_scale middle_ --track_buffer 40 --conf_thres 0.45 --nms_thres 0.7 --show_image True "
           "--challenge_name mot20-_all-test --track_eval_dir None --id_match_thres 0.45")

    import time
    wait = 2 * 3600
    for i in range(wait):
        print(f"\rstart track after {wait - i}s", end="")
        time.sleep(1)
    run_cmd(cmd)

