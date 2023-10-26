import torch
from torch import nn
from torch.nn import Module
import copy


class MemoryNet(Module):
    def __init__(self, in_channels):
        super().__init__()
        self.hl = nn.Linear(in_channels * 2, in_channels)
        self.zl = nn.Linear(in_channels * 2, in_channels)
        self.rl = nn.Linear(in_channels * 2, in_channels)
        self.features_dict = {}

    def forward(self, ids, features):
        old_features = []
        new_features = []
        up_id = []
        ids = ids.to('cpu').numpy() if isinstance(ids, torch.Tensor) else ids
        ids = [str(id) for id in ids]
        for id, feature in zip(ids, features):
            if id in self.features_dict:
                old_features.append(self.features_dict[id].view(1, -1))
                new_features.append(feature.view(1, -1))
                up_id.append(id)
            else:
                self.features_dict[str(id)] = feature.view(1, -1).detach().clone()

        if len(old_features) > 0:
            old_features = torch.cat(old_features, dim=0)
            new_features = torch.cat(new_features, dim=0)
            updated = self.linear_gru(old_features, new_features)
            for id, feature in zip(up_id, updated):
                self.features_dict[id] = feature.view(1, -1).detach().clone()
                ind = ids.index(id)
                features[ind] = feature

        return features

    def test_mode(self, old_feature, feature: torch.Tensor):
        feature = feature.view(1, -1)
        return self.linear_gru(old_feature, feature)

    def linear_gru(self, old_feature, feature):
        # print(old_feature.shape, feature.shape)
        cat = torch.cat((old_feature, feature), dim=1)

        # reset gate
        z = torch.sigmoid(self.zl(cat))

        # update gate
        r = torch.sigmoid(self.rl(cat))
        h = torch.tanh(self.hl(torch.cat((r * old_feature, feature), dim=1)))

        update = (1 - z) * feature + z * h
        return update

    def clear_dict(self):
        self.features_dict = {}

    def del_memory(self, id):
        if str(id) in self.features_dict:
            del self.features_dict[str(id)]