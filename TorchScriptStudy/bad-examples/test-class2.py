import torch

@torch.jit.script
class MyClass(object):
   name = "MyClass"
   def __init__(self, x: int):
      self.x = x

def fn(a: MyClass):
    return a.name

x = MyClass(1)
print("Eager: ", fn(x))
scripted_fn = torch.jit.script(fn)
print("Scripted: ", scripted_fn(x))

