import torch


def save_model(model, path):
    torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
