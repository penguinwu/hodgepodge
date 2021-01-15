import torch

from typing import Tuple
from typing import Any

@torch.jit.export
def incFirstElement(x: Tuple[int, Any]):
     return (x[0]+1, x[1])

m = torch.jit.script(incFirstElement)
print(incFirstElement((1,2.0)))
print(incFirstElement((1,(100,200))))

