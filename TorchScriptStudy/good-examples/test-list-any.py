import torch
from typing import Any
from typing import List

def appendToAnyList(a : List[Any], b: Any, c: Any):
    a.append(b)
    a.append(c)
    return a.sort()

#a = []
#a = appendToAnyList(a, 1, "hello")
#print("eager:", a)

m = torch.jit.script(appendToAnyList)
a = []
a = m(a, 1, "hello")
print("TorchScript:", a)

