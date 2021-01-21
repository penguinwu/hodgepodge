import torch
from typing import Tuple

#@torch.jit.script
class MyClass():
    def __init__(self, v: int, other: Tuple[int, int]):
        self.x = v
        self.other = other

    @torch.jit.export
    def add(self, y: int):
        return self.x + y

m = torch.jit.script(MyClass(2, (1,2)))
print(m.add(3))
