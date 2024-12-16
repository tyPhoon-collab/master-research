import torch


def auto_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
