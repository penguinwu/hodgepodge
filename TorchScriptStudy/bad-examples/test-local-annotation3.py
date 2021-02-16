import torch

def f(a: int, b: float, flag: bool):
    value: Any = b
    if flag:
        value = a
    return value

m = torch.jit.script(f)
print("Eager:", f(2, 1.0, True))
print("TorchScript:", m(2, 1.0, True))

