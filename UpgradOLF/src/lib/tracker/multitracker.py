import itertools
import os
import os.path as osp
import time
from collections import deque
from torchvision.ops import nms
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.lib.models.decode import mot_decode
from src.lib.models.model import create_model, load_model
from src.lib.models.utils import _tranpose_and_gather_feat, _tranpose_and_gather_feat_by_radius
from src.lib.tracking_utils.kalman_filter import KalmanFilter
from src.lib.tracking_utils.log import logger
from src.lib.tracking_utils.utils import *
from src.lib.utils.image import get_affine_transform
from src.lib.utils.post_process import ctdet_post_process

from src.lib.tracker import matching

from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.5      # golden ratio

    def update_features(self, feat, memory_net=None):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if memory_net is None:
                self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            else:
                with torch.no_grad():
                    feat = torch.tensor(feat)
                    memory_net = memory_net.to('cpu')
                    old_feature = torch.tensor(self.smooth_feat).view(1, -1)
                    self.smooth_feat = memory_net.test_mode(old_feature, feat).detach().numpy()
                    self.smooth_feat = self.smooth_feat.reshape((-1))

        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, memory_net=None):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat, memory_net)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True, memory_net=None):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat, memory_net)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def xyah_to_tlwh(xyah):
        ret = np.asarray(xyah).copy()
        ret[2] *= ret[3]
        ret[2] -= ret[2:] / 2.
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch_scale, opt.arch, opt.heads, opt.head_conv, pretrained=False)

        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()
        # self.model.train()
        self.init_tracks()

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])   # n, K, 4 + 1 + 1
        dets = ctdet_post_process(
            dets, [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections, id_features=None):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
                if id_features is not None:
                    id_features[j] = id_features[j][keep_inds]
        return results, id_features

    def init_tracks(self):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

    def update(self, im_blob, img0, is_new_sequence=False, input_type='video'):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            if is_new_sequence:
                self.model.new_sequence()
                self.init_tracks()
                STrack.restart_count()

            output = self.model(im_blob, input_type, purpose="for_test")[-1]
            memory_net = getattr(self.model, "memory_net") if hasattr(self.model, "memory_net") else None
            memory_net = None
            if memory_net is not None and is_new_sequence:
                memory_net.clear_dict()

            hm = output['hm'].sigmoid_()
            wh = output['wh']   # n, 2, h, w
            id_feature = output['id']   # n, 128, h, w
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            # dets: n, K, 4 + 1 + 1
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)

            # from src.lib.tracking_utils.component_visual import visual_detection
            # import copy
            # plot_img = np.ascontiguousarray(im_blob[0].cpu().permute(1, 2, 0).numpy())
            # print(im_blob.shape)
            # ratio = min(float(im_blob.shape[2]) / height, float(im_blob.shape[3]) / width)
            # new_shape = (round(height * ratio), round(width * ratio))  # new_shape = [width, height]
            # # print(new_shape, height, width)
            # dw = (im_blob.shape[3] - new_shape[1]) / 2  # width padding
            # dh = (im_blob.shape[2] - new_shape[0]) / 2  # height padding
            # top, bottom = round(dh - 0.1), round(dh + 0.1)
            # left, right = round(dw - 0.1), round(dw + 0.1)
            # vdets = copy.deepcopy(dets)
            # vdets[:, :, [0, 2]] = vdets[:, :, [0, 2]] / ratio - int(left / 4 / ratio)
            # vdets[:, :, [1, 3]] = vdets[:, :, [1, 3]] / ratio - int(top / 4 / ratio)
            # visual_detection(copy.deepcopy(img0), vdets, hm, 4, save_image=True,
            #                  save_dir=r"D:\BanYanDeng\expriment_figures", save_name="detections_4")
            # id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = _tranpose_and_gather_feat_by_radius(id_feature, inds, 1)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu()

        dets = dets.flatten(0, 1)[None]     # 1, n * K, 6
        id_feature = id_feature.flatten(0, 1).numpy()
        dets = self.post_process(dets, meta)
        id_feature = [id_feature for _ in range(len(dets.keys()) + 1)]
        dets, id_feature = self.merge_outputs([dets], id_feature)
        dets, id_feature = dets[1], id_feature[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = torch.tensor(dets[remain_inds])
        id_feature = id_feature[remain_inds]

        # nms
        remain_inds = nms(dets[:, :4], dets[:, 4], self.opt.nms_thres)
        dets = dets[remain_inds].numpy().reshape((-1, 5))
        id_feature = id_feature[remain_inds].reshape((-1, self.opt.reid_dim))
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, self.opt.track_buffer,
                                 ) for (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding distance and kalman predict'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        STrack.multi_predict(strack_pool)
        if len(detections) > 0:
            dists = matching.embedding_distance(strack_pool, detections)
            dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections,
                                         lambda_=self.opt.kalman_lambda)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.id_match_thres)
            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False, memory_net=memory_net)
                    refind_stracks.append(track)

            ''' Step 3: Second association, with IOU'''
            detections = [detections[i] for i in u_detection]
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.iou_match_thres)

            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id, memory_net=memory_net)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False, memory_net=memory_net)
                    refind_stracks.append(track)

            '''use kalman predict as not matched activate track or mark lost'''

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
                else:
                    track.frame_id = self.frame_id
                    track.tracklet_len += 1
                    track.mean, track.covariance = self.kalman_filter.update(
                        track.mean, track.covariance, track.tlwh_to_xyah(track._tlwh))

            '''step 3 Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed, detections)
            matches, u_unconfirmed, u_detection = (
                matching.linear_assignment(dists, thresh=self.opt.iou_match_for_unconfirmed_thres))

            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id, memory_net=memory_net)
                activated_starcks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

            """ Step 4: Init new stracks"""
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.det_thresh:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

            """ Step 5: Update state"""
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        if memory_net is not None:
            for track in self.removed_stracks:
                memory_net.del_memory(track.track_id)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


