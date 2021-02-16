import torch


@torch.jit.script
class MyClass(object):
   def __init__(self, x: int):
      self.x = x

   def set(self, val: int):
       self.x : float = val

def fn(a: MyClass, b: int):
    a.set(b)
    return a.x

x = MyClass(1)
print("Eager: ", fn(x, 100))

x = MyClass(1)
scripted_fn = torch.jit.script(fn)
print("Scripted: ", scripted_fn(x, 100))

