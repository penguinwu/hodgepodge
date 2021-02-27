import torch

def f(a : complex):
    print(a)
    return (isinstance(a, torch.Tensor))

ones = complex('1+2j')
m = torch.jit.script(f)
print("Eager:", f(ones))
print("TorchScript:", m(ones))

