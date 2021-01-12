import torch

class TestModule(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.x = v

    def mul(self, x: int):
        return self.x * x

    # compute self.x * x + y
    @torch.jit.export
    def madd(self, x: int, y: int):
        return self.mul(x) + y

    def forward(self, x: int):
        return self.x + x

m = torch.jit.script(TestModule(2))
print(m.madd(2, 3))

m = torch.jit.script(TestModule(2.0))
print(m.madd(2, 3))

m = torch.jit.script(TestModule(2.0))
print(m(2))
