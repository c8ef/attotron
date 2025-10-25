import builtins
import fcntl
import random

import numpy as np
import torch


def print(*args, is_print_rank=True, **kwargs):
    if not is_print_rank:
        return
    with open(__file__) as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def readable(num, precision=3):
    num_str = str(num)
    length = len(num_str)

    def format_with_precision(main, decimal, suffix):
        if precision == 0:
            return f"{main}{suffix}"
        return f"{main}.{decimal[:precision]}{suffix}"

    if length > 12:
        return format_with_precision(num_str[:-12], num_str[-12:], "T")
    elif length > 9:
        return format_with_precision(num_str[:-9], num_str[-9:], "B")
    elif length > 6:
        return format_with_precision(num_str[:-6], num_str[-6:], "M")
    elif length > 3:
        return format_with_precision(num_str[:-3], num_str[-3:], "K")
    else:
        return num_str
