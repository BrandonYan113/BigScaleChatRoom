from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _tranpose_and_gather_feat_by_radius(feat, ind, radius):
    '''

    :param feat: n, c, h, w
    :param ind:     n, K
    :param radius: int
    :return:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()    # n, h, w, c
    h, w = feat.shape[1:3]
    K = ind.shape[1]
    feat = feat.view(feat.size(0), -1, feat.size(3))
    inds, keeps = [], []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            k = ind + w * i + j
            y = torch.floor(ind / w)
            x = ind - y * w
            y += j
            x += i
            keep = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            inds.append(k)
            keeps.append(keep)

    inds = torch.stack(inds, dim=0).permute(1, 0, 2)  # n, (2 * radius + 1) ^ 2, K
    keeps = torch.stack(keeps, dim=0).permute(1, 0, 2)
    num = keeps.sum(dim=1).unsqueeze(2)
    assert (num > 0).all(), "input ind may be illegal"
    inds = torch.clip(inds, 0, w * h - 1).flatten(1, 2)
    feat = _gather_feat(feat, inds).unflatten(1, (-1, K))      # n, (2 * radius + 1) ^ 2, K, c
    feat = (feat * keeps.unsqueeze(3)).sum(dim=1) / num  # n, K, c
    return feat


def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)


def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2,
                      tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)
