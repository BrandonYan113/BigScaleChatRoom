cd src
python train.py --task mot --exp_id crowdhuman_middle_arch --arch_scale middle --gpus 0 --batch_size 5 --pretrained True --load_model ..\exp\mot\crowdhuman_middle_arch/model_10.pth --num_epochs 40 --lr_step 15,30 --data_cfg ../src/lib/cfg/crowdhuman.json --level_num 2 --conf_thres 0.35 --shuffle_every_epoch True --wh_weight 0.5
cd ..
