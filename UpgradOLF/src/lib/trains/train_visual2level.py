import copy
import cv2
import numpy as np
import torch
from src.utils.Decoder import ltrb_box_decode
from src.utils.plots import Annotator
from src.utils.visualization import colorize_heatmap
from src.trackerdemo.post_process import ctdet_post_process
from src.utils.embedding_utils import _tranpose_and_gather_feat
from torch.nn import functional as F


def train_visual(img0, dethm, detbox, gthm, gtbox, gtinds, ids, level_num, down_ratio, conf_thres, wait=100):
    gtdets = ltrb_box_decode(gtbox, gtinds, dethm.shape[-1]) * down_ratio    # n, K, 4
    gtscores = _tranpose_and_gather_feat(gthm, gtinds)
    gtdets = torch.cat((gtdets, gtscores), dim=2)
    gtimg = img0[None].repeat(level_num, 1, 1, 1).permute(0, 2, 3, 1).to("cpu").detach().numpy()
    gtimg = np.ascontiguousarray(gtimg)
    detimg = copy.deepcopy(gtimg)

    gt_result, gthm = plot(gtimg, gtdets, gthm, ids, down_ratio, conf_thres, (0, 255, 0))
    cv2.imshow("gt detects", gt_result)
    cv2.imshow("gt heatmap", gthm)
    if dethm is not None and detbox is not None:
        detbox = _tranpose_and_gather_feat(detbox, gtinds)
        detbox = ltrb_box_decode(detbox, gtinds, dethm.shape[-1]) * down_ratio
        detscores = _tranpose_and_gather_feat(dethm, gtinds)
        detbox = torch.cat((detbox, detscores), dim=2)
        det_result, hm = plot(detimg, detbox, dethm, None, down_ratio, conf_thres, (0, 0, 255))

        cv2.imshow("detects", det_result)
        cv2.imshow("det heatmap", hm)
        if cv2.waitKey(wait) & 0xff == 27:
            cv2.destroyAllWindows()


def plot(img0, dets, heat_maps, ids, down_ratio, conf_thres, color, pad_value=1.):
    show_img = []
    show_hm = []
    imgh, imgw = img0.shape[1:3]
    for i in range(2):
        remain_inds = dets[i][:, 4] > conf_thres
        det = dets[i][remain_inds]
        target_num = len(det)
        ann = Annotator(img0[i])
        for j, box in enumerate(det):
            ann.box_label(box[:-1], color=color, label=str(int(ids[i][j].item())) if ids is not None else "",
                          txt_color=(255, 255, 255))

        img = ann.result()
        text_scale = max(1, imgw / 1200.)
        cv2.putText(img, 'target_nums: %d' % (target_num),
                    (10, int(20 * text_scale)), cv2.FONT_HERSHEY_COMPLEX, text_scale, (0, 0, 255), thickness=1)

        show_img.append(img)
        heat_map = F.interpolate(heat_maps[i][None], scale_factor=down_ratio / 2.)
        heat_map = F.pad(heat_map, pad=(5, 5, 5, 5), value=pad_value)
        show_hm.append(heat_map)

    show_img = np.concatenate(show_img, axis=1)
    show_img = cv2.resize(show_img, (int(imgw * 1.25), int(imgh / 2 * 1.25)))

    show_hm = torch.cat(show_hm, dim=-1)
    show_hm = colorize_heatmap(show_hm)
    return show_img, show_hm


def gt_decode(heat_map, wh, gtinds, reg, ltrb):
    scores = _tranpose_and_gather_feat(heat_map, gtinds)

    K = wh.shape[1]
    height, width = heat_map.shape[-2:]

    clses = torch.zeros_like(gtinds)

    topk_inds = gtinds % (height * width)
    ys = torch.true_divide(topk_inds, width).int().float()
    xs = (topk_inds % width).int().float()
    if reg is not None:
        xs = xs.view(1, K, 1) + reg[:, :, 0:1]
        ys = ys.view(1, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(1, K, 1) + 0.5
        ys = ys.view(1, K, 1) + 0.5

    if ltrb:
        wh = wh.view(1, K, 4)
    else:
        wh = wh.view(1, K, 2)

    if ltrb:
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses[:, :, None]], dim=2)

    return detections


def post_process(dets, meta, num_classes):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])  # n, K, 4 + 1 + 1
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]


def merge_outputs(detections, num_classes, max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results
