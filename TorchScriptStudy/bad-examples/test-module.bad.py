import torch

class TestModule(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.x = v

    def forward(self, x: int):
        return self.x + x

class MyModel:
    def __init__(self, v: int):
        self.val = v

    @torch.jit.export
    def doSomething(self, val: int) -> int:
        # error: should not invoke the constructor of module type
        myModel = TestModule(self.val)
        return myModel(val)

print("Eager: ", MyModel(2).doSomething(3))

m = torch.jit.script(MyModel(2))
print("Scripted:", m.doSomething(3))
