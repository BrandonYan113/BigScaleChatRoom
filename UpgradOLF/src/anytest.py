from src.lib.opts import opts
from lib.models.utils import _tranpose_and_gather_feat_by_radius, _tranpose_and_gather_feat
import torch


if __name__ == '__main__':
    tensor = torch.randn((2, 4, 5, 5))
    ind = torch.tensor([[0, 4, 8],
           [1, 3, 7]], dtype=torch.int64)

    feat = _tranpose_and_gather_feat_by_radius(tensor, ind, 1)
    torch.set_printoptions(precision=4, sci_mode=False)
    print(tensor)
    print(feat)
    feat2 = _tranpose_and_gather_feat(tensor, ind)
    print(feat2.shape)