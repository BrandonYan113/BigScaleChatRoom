cd src
python train.py --task mot --exp_id mot17_20_half_middle_arch --arch_scale middle --gpus 0 --batch_size 5 --pretrained True --load_model ..\exp\mot\crowdhuman_middle_arch/model_last.pth --num_epochs 30 --lr_step 10,20 --data_cfg ../src/lib/cfg/mot17_20_half.json --level_num 2 --conf_thres 0.3 --id_weight 0.1 --wh_weight 0.1 --multi_loss None --lr 1e-4 --lr_gamma 0.5
cd ..