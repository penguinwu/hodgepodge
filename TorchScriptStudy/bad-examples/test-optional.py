import torch

def fn(a: int):
    b = None
    if a>100:
        b = 1
    return b

print("Eager: ", fn(101))
scripted_fn = torch.jit.script(fn)
print("Scripted: ", scripted_fn(101))

