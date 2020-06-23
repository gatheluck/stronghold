import torch


def save_model(model, path):
    torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), path)


def load_model(model, path):
    print('loading model weight from {path}'.format(path=path))

    # load weight from .pth file.
    if path.endswith('.pth'):
        model.load_state_dict(torch.load(path))
    # load weight from checkpoint.
    elif path.endswith('.ckpt'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise ValueError('this checkponint do not inculdes state_dict')
    else:
        raise ValueError('path is not supported type of extension.')
