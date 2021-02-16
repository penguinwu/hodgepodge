import torch

def f(a, setVal: bool):
    value: Optional[torch.Tensor] = None
    if setVal:
        value = a
    return value

ones = torch.ones([6])
m = torch.jit.script(f)
print("Eager:", f(ones, True), f(ones, False))
print("TorchScript:", m(ones, True), m(ones, False))
