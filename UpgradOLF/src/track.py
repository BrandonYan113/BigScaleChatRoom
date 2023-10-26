from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os.path as osp
import cv2
import logging
import motmetrics as mm
import numpy as np
import torch

from src.lib.tracker.multitracker import JDETracker
from src.lib.tracking_utils import visualization as vis
from src.lib.tracking_utils.log import logger
from src.lib.tracking_utils.timer import Timer
from src.lib.tracking_utils.evaluation import Evaluator, TrackEval
import src.lib.datasets.dataset.jde as datasets

from src.lib.tracking_utils.utils import mkdir_if_missing
from src.lib.opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)

    if save_dir == "None":
        save_dir = None
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    old_dir, cur_dir = "", ""
    for i, (path, img, img0) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        cur_dir = osp.dirname(path)
        is_new_sequence = False
        if old_dir != cur_dir:
            print(cur_dir, old_dir)
            is_new_sequence = True

        old_dir = cur_dir
        online_targets = tracker.update(blob, img0, is_new_sequence, opt.input_type)
        online_tlwhs = []
        online_ids = []
        #online_scores = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)

        timer.toc()

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
            if cv2.waitKey(1) & 0xff == 27:
                cv2.destroyAllWindows()

        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.bmp'.format(frame_id)), online_im)

        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', is_train_set=False, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True, track_eval_dir=None, challenge_name=None):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    track_eval_dir = None if track_eval_dir == "None" else track_eval_dir
    # save_images = False if save_images == "
    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    if track_eval_dir is not None:
        # write seqmaps txt
        seqmaps = os.path.join(track_eval_dir, "data/gt/mot_challenge", "seqmaps")
        if not os.path.exists(seqmaps):
            os.mkdir(seqmaps)
        seqmap_txt = os.path.join(seqmaps, challenge_name + ".txt")
        seqmap_file = open(seqmap_txt, 'w')
        seqmap_file.write(f"name\n")
        suffix = str(challenge_name).split("-")[-1]

    for seq in seqs:
        if track_eval_dir is not None:
            seqmap_file.write(seq + f"_{suffix}\n")

        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq.replace("\\", "/").split("/")[-1]))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename, save_dir=output_dir,
                              show_image=show_image, frame_rate=frame_rate,
                              use_cuda=True if opt.gpus[0] >= 0 else False)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)

        acc, keep_trks = evaluator.eval_file(result_filename)
        accs.append(acc)
        if track_eval_dir is not None:
            gt_root = os.path.join(f"{opt.data_dir}", data_root, seq)
            track_eval = TrackEval(track_eval_dir, challenge_name, exp_name)
            track_eval.eval_seq(seq, keep_trks, gt_root)

        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)

    if track_eval_dir is not None:
        seqmap_file.close()

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    if track_eval_dir is not None:
        track_eval.sys_do_eval()
        track_eval.print_keys_value(["HOTA", "MOTA", "MOTP", "IDF1", "DetA", "IDs", "GT_IDs", "AssA"])

    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    is_train_set = False
    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
        is_train_set = True

    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_danceTrack1:
        seqs_str = '''  dancetrack0003
                        dancetrack0009
                        dancetrack0011
                        dancetrack0013
                        dancetrack0017
                        dancetrack0021
                        dancetrack0022
                        dancetrack0028
                        dancetrack0031
                        dancetrack0036
                        dancetrack0038
                        dancetrack0040
                        dancetrack0042
                        dancetrack0046
                        dancetrack0048
                        dancetrack0050
                        dancetrack0054
                        dancetrack0056
                        dancetrack0059
                        dancetrack0060 '''
        data_root = os.path.join(opt.data_dir, 'test1')

    if opt.test_danceTrack2:
        seqs_str = '''
                    dancetrack0064
                    dancetrack0067
                    dancetrack0070
                    dancetrack0071
                    dancetrack0076
                    dancetrack0078
                    dancetrack0084
                    dancetrack0085
                    dancetrack0088
                    dancetrack0089
                    dancetrack0091
                    dancetrack0092
                    dancetrack0093
                    dancetrack0095
                    dancetrack0100
                        '''
        data_root = os.path.join(opt.data_dir, 'test2')

    if opt.val_danceTrack:
        seqs_str = '''
                    dancetrack0004
                    dancetrack0005
                    dancetrack0007
                    dancetrack0010
                    dancetrack0014
                    dancetrack0018
                    dancetrack0019
                    dancetrack0025
                    dancetrack0026
                    dancetrack0030
                    dancetrack0034
                    dancetrack0035
                    dancetrack0041
                    dancetrack0043
                    dancetrack0047
                    dancetrack0058
                    dancetrack0063
                    dancetrack0065
                    dancetrack0073
                    dancetrack0077
                    dancetrack0079
                    dancetrack0081
                    dancetrack0090
                    dancetrack0094
                    dancetrack0097
                    '''
        data_root = os.path.join(opt.data_dir, 'val')

    if opt.train_mot17:
        seqs_str = '''
                    train/MOT17-02-SDP
                    train/MOT17-04-SDP
                    train/MOT17-05-SDP
                    train/MOT17-09-SDP
                    train/MOT17-10-SDP
                    train/MOT17-11-SDP
                    train/MOT17-13-SDP
                    '''
        data_root = os.path.join(opt.data_dir, "MOT17")
    if opt.test_mot17_all:
        seqs_str = '''
                      test/MOT17-01-SDP
                      train/MOT17-02-SDP
                      test/MOT17-03-SDP
                      train/MOT17-04-SDP
                      train/MOT17-05-SDP
                      test/MOT17-06-SDP
                      test/MOT17-07-SDP
                      test/MOT17-08-SDP
                      train/MOT17-09-SDP
                      train/MOT17-10-SDP
                      train/MOT17-11-SDP
                      test/MOT17-12-SDP
                      train/MOT17-13-SDP
                      test/MOT17-14-SDP
                      test/MOT17-01-DPM
                      train/MOT17-02-DPM
                      test/MOT17-03-DPM
                      train/MOT17-04-DPM
                      train/MOT17-05-DPM
                      test/MOT17-06-DPM
                      test/MOT17-07-DPM
                      test/MOT17-08-DPM
                      train/MOT17-09-DPM
                      train/MOT17-10-DPM
                      train/MOT17-11-DPM
                      test/MOT17-12-DPM
                      train/MOT17-13-DPM
                      test/MOT17-14-DPM
                      test/MOT17-01-FRCNN
                      train/MOT17-02-FRCNN
                      test/MOT17-03-FRCNN
                      train/MOT17-04-FRCNN
                      train/MOT17-05-FRCNN
                      test/MOT17-06-FRCNN
                      test/MOT17-07-FRCNN
                      test/MOT17-08-FRCNN
                      train/MOT17-09-FRCNN
                      train/MOT17-10-FRCNN
                      train/MOT17-11-FRCNN
                      test/MOT17-12-FRCNN
                      train/MOT17-13-FRCNN
                      test/MOT17-14-FRCNN
                      '''
        data_root = os.path.join(opt.data_dir, "MOT17")

    if opt.test_mot17:
        seqs_str = '''MOT17-07-SDP
                      MOT17-01-SDP
                      MOT17-14-SDP
                      MOT17-06-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-03-SDP'''

        data_root = os.path.join(opt.data_dir, 'MOT17/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        # seqs_str = '''
        #             MOT17-02-SDP
        #             '''
        data_root = os.path.join(opt.data_dir, 'MOT17SDP/val')
        is_train_set = True

    if opt.test_mot20_all:
        seqs_str = '''train/MOT20-01
                      train/MOT20-02
                      train/MOT20-03
                      test/MOT20-04
                      train/MOT20-05
                      test/MOT20-06
                      test/MOT20-07
                      test/MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/val')
        is_train_set = True

    if opt.test_mot20:
        seqs_str = '''MOT20-07
                      MOT20-06
                      MOT20-04
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         is_train_set=is_train_set,
         seqs=seqs,
         exp_name=opt.exp_id + "_track",
         show_image=opt.show_image,
         save_images=opt.save_image,
         save_videos=False,
         track_eval_dir=opt.track_eval_dir,
         challenge_name=opt.challenge_name)
