import torch
from typing import Optional

@torch.jit.script
class MyClass(object):
    x: Optional[int]

    def __init__(self, x: int):
        self.x = None
        if x > 0:
            self.x = x

def fn(a: MyClass):
    return a.x

# x = MyClass(1)
# y = MyClass(0)
# print("Eager: ", fn(x), fn(y))

scripted_fn = torch.jit.script(fn)
print("Scripted: ", scripted_fn(MyClass(1)))
print("Scripted: ", scripted_fn(MyClass(0)))
