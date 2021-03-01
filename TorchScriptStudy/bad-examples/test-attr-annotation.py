import torch
from typing import Optional

@torch.jit.script
class MyClass(object):
    x: Optional[int]

    def __init__(self, x: int):
       self.x = x

    def inc(self, val: int):
        y: Optional[int] = val
        if y != None:
            return val
        if self.x != None:
            self.x += val

def fn(a: MyClass, b: int):
    a.inc(b)
    return a.x

x = MyClass(1)
y = MyClass(None)
print("Eager: ", fn(x, 100), fn(y, 100))

x = MyClass(1)
y = MyClass(None)
scripted_fn = torch.jit.script(fn)
print("Scripted: ", scripted_fn(x, 100), scripted_fn(y, 100))
