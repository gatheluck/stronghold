import os
import sys

import torch
from torch.autograd.gradcheck import zero_gradients

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(base)


def normalize_and_adjust(x, ratio: float, device: str):
    """
    Args
    - x (torch.tesnor): input tensor whos shape should be (b,c,h,w).
    """
    assert len(x.size()) == 4, "shape of x should be (b,c,h,w)."
    assert ratio > 0.0, "ratio should be larger than zero."

    # normalize
    std = x.std().item()
    x = torch.clamp(x, min=-ratio * std, max=ratio * std) / (ratio * std)

    # adjust
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(device)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(device)

    x.mul_(std[None, :, None, None]).add_(mean[None, :, None, None])
    return x


def normalize(x, mean, std, device: str):
    assert len(x.size()) == 4, "shape of x should be [b,c,h,w]"
    assert len(mean) == len(std), "channel size should be same."
    assert x.size(1) == len(mean), "channel size of x and mean/std should be same."

    mean, std = (
        torch.tensor(mean, dtype=torch.float32).to(device),
        torch.tensor(std, dtype=torch.float32).to(device),
    )
    return x.sub(mean[None, :, None, None]).div(std[None, :, None, None])


def unnormalize(x, mean, std, device: str):
    assert len(x.size()) == 4, "shape of x should be [b,c,h,w]"
    assert len(mean) == len(std), "channel size should be same."
    assert x.size(1) == len(mean), "channel size of x and mean/std should be same."

    mean, std = (
        torch.tensor(mean, dtype=torch.float32).to(device),
        torch.tensor(std, dtype=torch.float32).to(device),
    )
    return x.mul(std[None, :, None, None]).add(mean[None, :, None, None])


class VanillaGrad:
    def __init__(self, model, num_classes: int, device: int):
        self.num_classes = num_classes
        self.device = device
        self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, x, t):
        x, t = x.to(self.device), t.to(self.device)
        x.requires_grad = True
        logit = self.model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1))
        onehot_t = torch.nn.functional.one_hot(t, num_classes=self.num_classes).float()

        logit.backward(gradient=onehot_t)
        grad = x.grad.data

        return grad.detach()


class LossGrad:
    def __init__(
        self,
        model,
        num_classes: int,
        device: str,
        criterion=torch.nn.CrossEntropyLoss(),
    ):
        self.num_classes = num_classes
        self.device = device
        self.criterion = criterion
        self.model = model.to(self.device)
        self.model.eval()

    def __call__(self, x, t):
        x, t = x.to(self.device), t.to(self.device)
        x.requires_grad = True
        logit = self.model(x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1))
        loss = self.criterion(logit, t)

        zero_gradients(x)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        grad = x.grad.data

        return grad.detach()


if __name__ == "__main__":
    import PIL
    import torchvision

    num_classes = 100
    device = "cuda"
    weight_path = os.path.join("../samples", "clean_model_weight.pth")
    # weight_path = os.path.join("../samples", 'pgd-linf-8_model_weight.pth')
    model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path))

    img = PIL.Image.open("../samples/ILSVRC2012_val_00000466.JPEG")
    t = torch.tensor([1])  # black swan
    mean = (0.485, 0.456, 0.406)  # mean of ImageNet100
    std = (0.229, 0.224, 0.225)  # std of ImageNet100

    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )

    x = preprocess(img)[None, :, :, :].to(device)
    x = normalize(x, mean, std, device)

    visualizers = list()
    visualizers.append(VanillaGrad(model, num_classes, device))
    visualizers.append(LossGrad(model, num_classes, device))

    for visualizer in visualizers:
        grad = visualizer(x, t)
        x_cat = torch.cat(
            [unnormalize(x, mean, std, device), normalize_and_adjust(grad, 5.0, device)]
        )
        torchvision.utils.save_image(
            x_cat.detach(),
            os.path.join(
                "../logs", "result_{vis}.png".format(vis=visualizer.__class__.__name__)
            ),
            padding=0,
        )
