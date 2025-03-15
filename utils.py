import torch

def tensor2parameter(x):
    if isinstance(x, torch.nn.Parameter):
        return x
    else:
        return torch.nn.Parameter(x, requires_grad=False)

def soft_empty_cache(force=False):
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
    if torch.cuda.is_available():
        if torch.version.cuda:  # This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
