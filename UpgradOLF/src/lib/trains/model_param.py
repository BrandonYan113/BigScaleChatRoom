from torch import nn
import numpy as np
import torch


def set_parameter(keys: list, model: nn.Module, lrs=0.01, weight_decays=1e-8):
    if keys[0] == 'all':
        if isinstance(lrs, list):
            lrs = lrs[0]
        if isinstance(weight_decays, list):
            weight_decays = weight_decays[0]

        params = [param for param in model.parameters() if param.requires_grad]

        return [{"params": params, 'lr': lrs, 'weight_decay': weight_decays}]
    else:
        params = {str(key): [] for key in keys}
        for name, param in model.named_parameters():
            for key in keys:
                if name.startswith(key) and param.requires_grad:
                    params[str(key)].append(param)

        flag = np.array([len(param) for param in params.values()])
        not_found = flag == 0
        if not (flag > 0).all():
            error = f"not found keys: {np.array(keys)[not_found]}"
            raise ValueError(error)

        if isinstance(lrs, float):
            lrs = [lrs] * len(keys)
        if isinstance(weight_decays, float):
            weight_decays = [weight_decays] * len(keys)

        assert len(lrs) == len(keys), 'keys and lr are not matched'
        assert len(weight_decays) == len(keys), 'keys and lr are not matched'

        params_list = []
        for value, lr, weight_decay in zip(params.values(), lrs, weight_decays):
            dict = {"params": value, 'lr': lr, 'weight_decay': weight_decay}
            params_list.append(dict)

        return params_list


def load_state(model: nn.Module, loaded_state_dict):
    for name, pth in loaded_state_dict.items():
        try:
            if name == 'all':
                subnet = model
            else:
                subnet = getattr(model, name)
        except AttributeError as e:
            print(f'Error {e}')
        try:
            state_dict = torch.load(pth)['state_dict']
            # print(name)
            # print(state_dict["net"].keys())
            # exit(0)
            subnet.load_state_dict(state_dict=state_dict['net'], strict=True)
            print(f"subnet {name} state dict is reloaded")
        except FileNotFoundError:
            print(f"file {pth} not found")

