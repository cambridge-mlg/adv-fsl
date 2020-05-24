import torch


def convert_labels(predictions):
    return torch.argmax(predictions, dim=1, keepdim=False)
