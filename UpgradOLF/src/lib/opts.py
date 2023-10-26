from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--task', default='mot', help='mot')
        self.parser.add_argument('--dataset', default='jde', help='jde')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        # self.parser.add_argument('--load_model', default='../models/ctdet_coco_dla_2x.pth',
        # help='path to pretrained model')
        self.parser.add_argument("--pretrained", type=bool, default=False, help="load pretrained model")
        self.parser.add_argument('--load_model', default='..\exp\mot\mot17_20_middle_arch/model_30.pth',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=2023,
                                 help='random seed')  # from CornerNet

        # log
        self.parser.add_argument('--print_or_show_iter', type=int, default=100,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', type=bool, default=False,
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.5,
                                 help='visualization threshold.')

        # model
        self.parser.add_argument("--arch_scale", default="middle", type=str)
        self.parser.add_argument('--arch', default='dla_34',
                                 help='model architecture. Currently tested'
                                      'resdcn_34 | resdcn_50 | resfpndcn_34 |'
                                      'dla_34 | hrnet_18')
        self.parser.add_argument('--level_num', type=int, default=2)
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '256 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')

        # input
        self.parser.add_argument("--shuffle_every_epoch", type=bool, default=False)
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr', type=float, default=1e-4,
                                 help='learning rate for batch size 12.')
        self.parser.add_argument('--lr_step', type=str, default='20',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=30,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=4,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=6,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                 help='include validation in training and '
                                      'test on test set')

        # test
        self.parser.add_argument("--flush_frq", type=int, default=10)
        self.parser.add_argument("--imitate_box", type=bool, default=False,
                                 help="imitate model output with gt file")
        self.parser.add_argument('--K', type=int, default=500,
                                 help='max number of output objects.')
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')
        # tracking
        self.parser.add_argument('--test_mot16', default=False, help='test mot16')
        self.parser.add_argument('--val_mot16', default=False, help='val mot16 or mot15')
        self.parser.add_argument("--test_mot17_all", default=False, help="include all data")
        self.parser.add_argument("--train_mot17", default=False)
        self.parser.add_argument('--test_mot17', default=False, help='test mot17')
        self.parser.add_argument('--val_mot17', default=False, help='val mot17')
        self.parser.add_argument('--test_mot20_all', default=False, help="include all data")
        self.parser.add_argument('--val_mot20', default=False, help='val mot20')
        self.parser.add_argument('--test_mot20', default=False, help='test mot20')
        self.parser.add_argument('--test_danceTrack1', default=False, help='test danceTrack test1')
        self.parser.add_argument('--test_danceTrack2', default=False, help='test danceTrack test2')
        self.parser.add_argument('--val_danceTrack', default=False, help='test danceTrack val')
        self.parser.add_argument('--val_hie', default=False, help='val hie')
        self.parser.add_argument('--test_hie', default=False, help='test hie')
        self.parser.add_argument('--conf_thres', type=float, default=0.4, help='confidence thresh for tracking')
        self.parser.add_argument('--nms_thres', type=float, default=0.7, help='iou thresh for nms')
        self.parser.add_argument('--kalman_lambda', type=float, default=0.02)
        self.parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
        self.parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
        self.parser.add_argument('--input-video', type=str,
                                 default='../videos/MOT16-03.mp4',
                                 help='path to the input video')
        self.parser.add_argument("--input_type", type=str, default="image", choices=['video', 'image'],
                                 help='video for video or image, image only for image detecting')
        self.parser.add_argument('--input_type_switch_frq', type=str, default="15",
                                 help="mode change from video to image or image to video")
        self.parser.add_argument('--output-format', type=str, default='video', help='video or text')
        self.parser.add_argument('--output-root', type=str, default='../demos', help='expected output root path')
        self.parser.add_argument("--save_image", type=bool, default=False, help="save detect result per image")
        self.parser.add_argument("--id_match_thres", type=float, default=0.4)
        self.parser.add_argument("--iou_match_thres", type=float, default=0.5)
        self.parser.add_argument("--iou_match_for_unconfirmed_thres", type=float, default=0.7)

        # eval
        self.parser.add_argument("--track_eval_dir", type=str, default="D:\A\TrackEval")
        self.parser.add_argument("--challenge_name", type=str, default="mot17-val",
                                 help="suffix must be [-val, -train, -test]")

        # mot
        self.parser.add_argument('--data_cfg', type=str,
                                 default='../src/lib/cfg/data.json',
                                 help='load data from cfg')
        self.parser.add_argument('--data_dir', type=str, default='D:\data')
        # self.parser.add_argument('--data_dir', type=str, default="D:\BanYanDeng\MOTDataset")

        # loss
        self.parser.add_argument('--mse_loss', action='store_true',
                                 help='use mse loss or focal loss to train '
                                      'keypoint heatmaps.')

        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')
        self.parser.add_argument('--id_loss', default='ce',
                                 help='reid loss: ce | focal')
        self.parser.add_argument('--id_weight', type=float, default=1,
                                 help='loss weight for id')
        self.parser.add_argument('--reid_dim', type=int, default=128,
                                 help='feature dim for reid')
        self.parser.add_argument('--ltrb', default=False,
                                 help='regress left, top, right, bottom of bbox')
        self.parser.add_argument('--multi_loss', type=str, default='uncertainty',
                                 help='multi_task loss: uncertainty | fix')

        self.parser.add_argument('--norm_wh', action='store_true',
                                 help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center or '
                                      'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh', action='store_true',
                                 help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='not regress local offset.')
        self.parser.add_argument('--show_image', default=False, help="show image when test")

        # warm up
        self.parser.add_argument('--warmup_epochs', type=int, default=0)
        self.parser.add_argument('--warmup_lr', type=float, default=1e-4)
        self.parser.add_argument('--warmup_keys', type=list, default=["association"])

        # lr scheduler
        self.parser.add_argument('--lr_gamma', type=float, default=0.1)
        self.parser.add_argument('--lr_last_epoch', type=int, default=-1)

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.input_type_switch_frq = [int(epoch) for epoch in opt.input_type_switch_frq.split(",")
                                     if "," in opt.input_type_switch_frq]
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',') if ',' in opt.lr_step]
        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 256
        opt.pad = 31
        opt.num_stacks = 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)

        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task == 'mot':
            opt.heads = {'hm': opt.num_classes,
                         'wh': 2 if not opt.ltrb else 4,
                         'id': opt.reid_dim}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
            opt.nID = dataset.nID if dataset.nID > 1 else 2
            opt.img_size = (1088, 608)
            # opt.img_size = (864, 480)
            # opt.img_size = (576, 320)
        else:
            assert 0, 'task not defined!'
        print('heads', opt.heads)

        return opt

    def init(self, args=''):
        default_dataset_info = {
            'mot': {'default_resolution': [608, 1088], 'num_classes': 1,
                    'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                    'dataset': 'jde', 'nID': 14455},
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
