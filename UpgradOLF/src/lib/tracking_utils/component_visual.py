import cv2
import torch
import os
from torch.nn import functional as F
from src.lib.trains.train_utils import colorize_heatmap
from src.lib.trains.train_visual import plot2level
from src.lib.models.decode import mot_decode
import copy
import numpy as np


def visual(high: torch.Tensor, low: torch.Tensor, format_type="abs_mean", scale=8,
           save_image=False, save_dir="../default", save_name="default", wait=True,
           inversed=True):
    assert high.dim() == 4, "input dim should be 4"
    assert low.dim() == 4, "input dim should be 4"

    high = F.interpolate(high.clone(), scale_factor=scale, mode="bicubic")
    low = F.interpolate(low.clone(), scale_factor=scale, mode="bicubic")
    high = colorize_heatmap(format_channel(high, format_type))
    low = colorize_heatmap(format_channel(low, format_type))
    if inversed:
        high = 1. - high
        low = 1. - low
    cv2.imshow(save_name + "_high", high)
    cv2.imshow(save_name + "_low", low)
    if wait:
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    if save_image:
        cv2.imwrite(os.path.join(save_dir, save_name + "_high.bmp"), high * 255.)
        cv2.imwrite(os.path.join(save_dir, save_name + "_low.bmp"), low * 255.)


def format_channel(x: torch.Tensor, type='mean'):
    if type == 'mean':
        return x.mean(dim=1, keepdim=True)
    elif type == 'abs_mean':
        return x.abs().mean(dim=1, keepdim=True)
    elif type == "sigmoid_mean":
        return torch.sigmoid(x.mean(dim=1, keepdim=True))
    elif type == "sigmoid_sum":
        return torch.sigmoid(x.sum(dim=1, keepdim=True))

    elif type == 'abs_sum':
        return x.abs().sum(dim=1, keepdim=True)
    elif type == "max_min_normal":
        x = (x - x.min()) / (x.max() - x.min())
        return x.max(dim=1, keepdim=True)[0]
    elif type == "max":
        return x.max(dim=1, keepdim=True)[0]
    elif type == "abs_max":
        return x.abs().max(dim=1, keepdim=True)[0]
    elif type == "gauss_normal":
        return torch.sum((x - x.mean()) / x.std(), dim=1, keepdim=True)
    else:
        msg = f"undefined format_type: {type}"
        raise ValueError(msg)


def visual_detection(detimg, dets, heat_map, scale, conf_thres=0.4, wait=True,
                     save_image=False, save_dir="../default", save_name="default"):
    
    if save_image:
        cv2.imwrite(os.path.join(save_dir, save_name + "_original.bmp"), detimg)
    det_result, hm = plot2level((detimg, copy.deepcopy(detimg)), dets, heat_map, None,
                                scale, conf_thres, (0, 0, 255))
    cv2.imshow(f"", det_result)
    if np.max(det_result) <= 1.:
        det_result = 255. * det_result
    if save_image:
        cv2.imwrite(os.path.join(save_dir, save_name + ".bmp"), det_result)
    if wait:
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()