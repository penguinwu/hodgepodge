import torch
from typing import Any

# To test auto-inference of return types
def f(a):
    print(a)

ones = torch.ones([6])
m = torch.jit.script(f)
print("Eager:", f(ones))
print("TorchScript:", m(ones))

