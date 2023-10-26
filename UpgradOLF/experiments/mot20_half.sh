cd src
python train.py --task mot --exp_id mot20_middle_arch --arch_scale middle --gpus 0 --batch_size 10 --pretrained True --load_model ..\exp\mot\crowdhuman_middle_arch/model_last.pth --num_epochs 30 --lr_step 25 --data_cfg ../src/lib/cfg/mot20_half.json --level_num 2 --conf_thres 0.3 --id_weight 0.1 --wh_weight 0.1 --multi_loss None --lr 1e-4 --input_type video
cd ..