import torch
from typing import Any

def f(a : Any):
    print(a)
    return (isinstance(a, torch.Tensor))

ones = torch.ones([6])
mine = torch.Tensor([1,2,3,4,5,6])
m = torch.jit.script(f)
print("Eager:", f(ones))
print("TorchScript:", m(ones))

