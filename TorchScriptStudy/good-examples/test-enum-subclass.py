import torch
from enum import Enum

class BaseColor(Enum):
    def foo(self):
        pass

class Color(BaseColor):
    RED = 1
    GREEN = 2

def enum_fn(x: Color, y: Color) -> bool:
    if x == Color.RED:
        return True

    return x == y

m = torch.jit.script(enum_fn)
print("TorchScript: ", m(Color.RED, Color.GREEN))
print("Eager: ", enum_fn(Color.RED, Color.GREEN))


