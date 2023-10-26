cd src
python train.py --task mot --exp_id crowdhuman_middle_arch --arch_scale middle --gpus 0 --batch_size 10 --pretrained True --load_model ..\src/lib/models/fairmot_dla34.pth --num_epochs 50 --lr_step 20,40 --data_cfg ../src/lib/cfg/crowdhuman.json --level_num 2 --conf_thres 0.3 --id_weight 0. --wh_weight 0.1 -input_type image --lr 1e-4
cd ..
