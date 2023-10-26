import copy
import cv2
import numpy as np
import torch
from src.lib.models.decode import mot_decode, _topk, _nms
from src.lib.trains.train_utils import Annotator, colorize_heatmap
from src.lib.utils.post_process import ctdet_post_process
from src.lib.models.utils import _tranpose_and_gather_feat
from torch.nn import functional as F
from torchvision.ops import nms


def plot_eval(batch, output, opt):
    index = np.random.randint(len(batch["input"]))
    img0 = batch["input"][index].detach()
    if opt.level_num == 1:
        gthm, gtwh = batch["hm"][index][None], batch["wh"][index][None]
        gtind = batch["ind"][index][None]
        gtreg = batch['reg'][index][None] if opt.reg_offset else None

        hm, wh = output["hm"][index][None], output["wh"][index][None]
        reg = output['reg'][index][None] if opt.reg_offset else None
        train_visual(img0, hm, wh, gthm, gtwh, reg, gtreg, gtind, opt.down_ratio, opt.ltrb,
                     opt.K, opt.conf_thres, opt.num_classes, img_index=0)

    elif opt.level_num == 2:
        gthm = batch['hm'].unflatten(0, (-1, opt.level_num))[index]
        gtwh = batch['wh'].unflatten(0, (-1, opt.level_num))[index]
        gtind = batch['ind'].unflatten(0, (-1, opt.level_num))[index]
        gtreg = batch['reg'].unflatten(0, (-1, opt.level_num))[index]
        # ids = batch['ids'].unflatten(0, (-1, opt.level_num))[index]

        dethm = output['hm'].unflatten(0, (-1, opt.level_num))[index]
        detwh = output['wh'].unflatten(0, (-1, opt.level_num))[index]
        detreg = output['reg'].unflatten(0, (-1, opt.level_num))[index] if \
            opt.reg_offset else None
        train_visual(img0, dethm, detwh, gthm, gtwh, detreg, gtreg, gtind, opt.down_ratio,
                     opt.ltrb, opt.K, opt.conf_thres, opt.num_classes, 2,
                     img_index=0)

        if index + 1 < len(batch["input"]):
            index_ = index + 1
        elif index - 1 >= 0:
            index_ = index - 1
        else:
            index_ = None

        if index_ is not None:
            img1 = copy.deepcopy(batch["input"][index_].detach())
            gthm_ = batch['hm'].unflatten(0, (-1, opt.level_num))[index_]
            gtwh_ = batch['wh'].unflatten(0, (-1, opt.level_num))[index_]
            gtind_ = batch['ind'].unflatten(0, (-1, opt.level_num))[index_]
            gtreg_ = batch['reg'].unflatten(0, (-1, opt.level_num))[index_]
            # ids = batch['ids'].unflatten(0, (-1, opt.level_num))[index]

            dethm_ = output['hm'].unflatten(0, (-1, opt.level_num))[index_]
            detwh_ = output['wh'].unflatten(0, (-1, opt.level_num))[index_]
            detreg_ = output['reg'].unflatten(0, (-1, opt.level_num))[index_] if \
                opt.reg_offset else None
            train_visual(img1, dethm_, detwh_, gthm_, gtwh_, detreg_, gtreg_, gtind_, opt.down_ratio,
                         opt.ltrb, opt.K, opt.conf_thres, opt.num_classes, 2,
                         img_index=1)
    else:
        msg = f'undefined level_num: {opt.level_num}, it should be 1 or 2'
        raise ValueError(msg)
    

def train_visual(img0, heat_map, wh, gthm, gtwh, reg, gtreg, gtinds, down_ratio,
                 ltrb, K, conf_thres, num_classes, level_num=1, img_index=0, wait=100, ids=None):
    dets, _ = mot_decode(heat_map, wh, reg=reg, ltrb=ltrb, K=K)
    gtdets = gt_decode(gthm, gtwh, gtinds, reg=gtreg, ltrb=ltrb, level_num=level_num, K=K)
    height, width = img0.shape[-2:]

    gtimg = np.ascontiguousarray(img0.permute(1, 2, 0).to("cpu").detach().numpy())
    detimg = copy.deepcopy(gtimg)

    if level_num == 1:
        gt_result, gthm = plot(gtimg, gtdets, gthm, down_ratio, conf_thres, (0, 255, 0))
        det_result, hm = plot(detimg, dets, heat_map, down_ratio, conf_thres, (0, 0, 255))
    elif level_num == 2:
        gt_result, gthm = plot2level((gtimg, copy.deepcopy(gtimg)), gtdets, gthm, ids,
                                     down_ratio, conf_thres, (0, 255, 0))
        det_result, hm = plot2level((detimg, copy.deepcopy(detimg)), dets, heat_map, None,
                                    down_ratio, conf_thres, (0, 0, 255))

    cv2.imshow(f"gt detects{img_index}", gt_result)
    cv2.imshow(f"gt heatmap{img_index}", gthm)
    cv2.imshow(f"detects{img_index}", det_result)
    cv2.imshow(f"det heatmap{img_index}", hm)
    if cv2.waitKey(wait) & 0xff == 27:
        cv2.destoryAllWindows()


def plot2level(img0, dets, heat_maps, ids, down_ratio, conf_thres, color, pad_value=1.):
    show_img = []
    show_hm = []
    imgh, imgw = img0[0].shape[:2]
    for i in range(2):
        remain_inds = dets[i][:, 4] > conf_thres
        det = dets[i][remain_inds]
        target_num = len(det)
        ann = Annotator(img0[i], line_width=2)
        for j, box in enumerate(det):
            ann.box_label(box[:-1] * down_ratio, color=color, label=str(int(ids[i][j].item())) if
            ids is not None else "", txt_color=(255, 255, 255))

        img = ann.result()
        text_scale = max(1, imgw / 1200.)
        cv2.putText(img, 'target_nums: %d' % (target_num),
                    (10, int(20 * text_scale)), cv2.FONT_HERSHEY_COMPLEX, text_scale, (0, 255, 0), thickness=1)

        show_img.append(img)
        heat_map = F.interpolate(heat_maps[i][None], scale_factor=down_ratio / 2.)
        heat_map = F.pad(heat_map, pad=(5, 5, 5, 5), value=pad_value)
        show_hm.append(heat_map)

    show_img = np.concatenate(show_img, axis=1)
    # show_img = cv2.resize(show_img, (int(imgw * 1.25), int(imgh / 2 * 1.25)))

    show_hm = torch.cat(show_hm, dim=-1)
    show_hm = colorize_heatmap(show_hm)
    return show_img, show_hm


def plot(img0, dets, heat_map, down_ratio, conf_thres, color):
    remain_inds = dets[0][:, 4] > conf_thres
    dets = dets[0][remain_inds]
    target_num = len(dets)
    ann = Annotator(img0)
    for det in dets:
        ann.box_label(det[:-2] * 4., color=color)

    show_img = ann.result()
    text_scale = max(1, show_img.shape[1] / 1600.)
    cv2.putText(show_img, 'target_nums: %d' % (target_num),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    heat_map = torch.nn.functional.interpolate(heat_map, scale_factor=down_ratio / 2.)
    heat_map = colorize_heatmap(heat_map)
    return show_img, heat_map


def gt_decode(heat_map, wh, gtinds, reg, ltrb, level_num=1, K=500):
    scores = _tranpose_and_gather_feat(heat_map, gtinds)

    height, width = heat_map.shape[-2:]

    clses = torch.zeros_like(gtinds)

    topk_inds = gtinds % (height * width)
    ys = torch.true_divide(topk_inds, width).int().float()
    xs = (topk_inds % width).int().float()
    if reg is not None:
        xs = xs.view(level_num, K, 1) + reg[:, :, 0:1]
        ys = ys.view(level_num, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(level_num, K, 1) + 0.5
        ys = ys.view(level_num, K, 1) + 0.5

    if ltrb:
        wh = wh.view(level_num, K, 4)
    else:
        wh = wh.view(level_num, K, 2)

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
