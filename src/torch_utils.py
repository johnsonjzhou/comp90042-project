"""
Use PyTorch on Apple Metal framework
Ref:
https://pytorch.org/docs/stable/notes/mps.html
"""
import torch
from src.sys_platform import is_apple_silicon

def get_mps_device() -> torch.device:
    """
    Checks Metal backend is available and retrieves the `mps` device.

    Raises:
        Exception: MPS is not available due to build.
        Exception: MPS is not available due to MacOS version.

    Returns:
        torch.device: A `mps` device that can be used for models or tensors.
    """
    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            raise Exception(
                """
                MPS not available because the current PyTorch install was not
                built with MPS enabled.
                """
            )
        else:
            raise Exception(
                """
                MPS not available because the current MacOS version is not 12.3+
                and/or you do not have an MPS-enabled device on this machine.
                """
            )
    else:
        print("MPS is available")
        mps_device = torch.device("mps")
        return mps_device

def get_torch_device() -> torch.device:
    """
    Gets a torch device based on priority:
    1. Metal Performance Shader (mps)
    2. CUDA GPU (cuda)
    3. CPU

    Returns:
        torch.device: A torch device.
    """
    if torch.backends.mps.is_available():
        print("Torch device is 'mps'")
        return torch.device("mps")

    if torch.cuda.is_available():
        print("Torch device is 'cuda'")
        return torch.device("cuda")

    return torch.device("cpu")
