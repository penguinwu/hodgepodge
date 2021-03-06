import torch

def f(a, b):
    # type: (torch.Tensor, int) -> torch.Tensor
    return a+b

ones = torch.ones([6])
m = torch.jit.script(f)
print("Eager:", f(ones, 100))
print("TorchScript:", m(ones, 100))

