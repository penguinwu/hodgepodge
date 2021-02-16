import torch

def f(a: int, b: float, flag: bool):
    value: Any = a if flag else b
    return value

m = torch.jit.script(f)
print("Eager:", f(1, 1.0, True), f(1, 1.0, False))
print("TorchScript:", m(1, 1.0, True), m(1, 1.0, False))

