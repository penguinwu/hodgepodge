import torch

def f(a, mask):
  if mask is not None:
    return a + mask
  return a

ones = torch.ones([6])
mine = torch.Tensor([1,2,3,4,5,6])
m = torch.jit.script(f)
print(m(ones, mine))
print(m(ones, None))

