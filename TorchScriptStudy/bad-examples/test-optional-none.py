import torch
from typing import Optional

def fn(a: Optional[int], val: int):
    if a != None:
        return a+val
    else:
        return val

x1 = None
x2 = 1
y = 1
print("Eager: ", fn(x1, y), fn(x2, y))

scripted_fn = torch.jit.script(fn)
print("Scripted: ", scripted_fn(x1, y), scripted_fn(x2, y))
