import torch
from typing import List

def f(append: bool, val: int):
    l = torch.jit.annotate(List[int], [])
    if append:
        l.append(val)
        return l
    else:
        return None

m = torch.jit.script(f)
print("Eager:", f(True, 1), f(False, 1))
print("TorchScript:", m(True, 1), m(False, 1))
