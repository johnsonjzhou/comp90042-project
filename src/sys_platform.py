"""
Get information about the system platform
"""

import os

def is_apple_silicon() -> bool:
    """
    Check if we are running on an Apple Silicon machine.

    Returns:
        bool: `True` if Apple Silicon.
    """
    sys_info = os.uname()
    if sys_info.sysname == "Darwin" and sys_info.machine == "arm64":
        return True
    return False