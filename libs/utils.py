import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import itertools
import torch


def parse_args(required_keys: set, input_args: dict) -> dict:
    """
    check if all required_keys are inculuded in input_args.
    return parsed_args only inculudes required keys.

    Args
    - required_keys (set) : set of required keys for input_args
    - input_args (dict)   : dict of input arugments
    """
    parsed_args = dict()

    for k in required_keys:
        if k not in input_args.keys():
            raise ValueError('initial args are invalid.')
        else:
            parsed_args[k] = input_args[k]

    return parsed_args


def get_epoch_end_log(outputs: list) -> dict:
    """
    this function fill the gap between single theread and data.parallel.
    form of outputs is List[Dict[str, Tensor]] or List[List[Dict[str, Tensor]]]
    """
    log = dict()

    # if list is nested, flatten them.
    if type(outputs[0]) is list:
        outputs = [x for x in itertools.chain(*outputs)]

    if 'log' in outputs[0].keys():
        print(outputs[0]['log'].keys())
        for key in outputs[0]['log'].keys():
            val = torch.stack([x['log'][key] for x in outputs]).mean().cpu().item()
            log[key + '_avg'] = val
    else:
        for key in outputs[0].keys():
            val = torch.stack([x[key] for x in outputs]).mean().cpu().item()
            log[key + '_avg'] = val

    return log


def freeze_params(model, unfreeze_param_names: list):
    """
    freeze params which are not inculuded in unfreeze_params.
    """
    for k, v in model.named_parameters():
        if k not in unfreeze_param_names:
            v.requires_grad = False

    return model


if __name__ == '__main__':
    import torchvision

    model = torchvision.models.resnet50()
    unfreeze_param_names = 'layer4.2.bn3.weight layer4.2.bn3.bias fc.weight fc.bias'.split()

    model = freeze_params(model, unfreeze_param_names)

    for k, v in model.named_parameters():
        print('{k}: {requires_grad}'.format(k=k, requires_grad=v.requires_grad))
