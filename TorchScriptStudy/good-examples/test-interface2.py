import torch

@torch.jit.interface
class MyInterface:
    def compute(self, val: int) -> int:
        return val

@torch.jit.script
class AddClass(object):
   def __init__(self, x: int):
      self.x = x

   def compute(self, val: int):
      self.x += val
      return self.x

@torch.jit.script
class MulClass(object):
   def __init__(self, x: int):
      self.x = x

   def compute(self, val: int):
      self.x *= val
      return self.x

def fn(a: MyInterface, b: int):
    return a.compute(b)

x = AddClass(2)
y = MulClass(2)
print("Eager: ", fn(x, 100))
print("Eager: ", fn(y, 100))

x = AddClass(2)
y = MulClass(2)
scripted_fn = torch.jit.script(fn)
print("Scripted: ", scripted_fn(x, 100))
print("Scripted: ", scripted_fn(y, 100))

