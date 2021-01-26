import torch

@torch.jit.script
class A():
    def __init__(self):
        self.x = torch.rand(3)

    def f(self, y: torch.device):
        return self.x.to(device=y)

def g():
    a = A()
    return a.f(torch.device("cpu"))

script_g = torch.jit.script(g)
print(script_g.graph)
