import torch
import pdb

# @torch.jit.script
class TestSubModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v = torch.rand((2, 3))

    def forward(self):
        return self.v

# @torch.jit.script
class MyTest():
    def __init__(self):
        self.x = 1

    def __str__(self):
        return "MyTest: " + str(self.x)

    def print(self):
        print("MyTest: ", self.x)

class TestModule(torch.nn.Module):
    # Comment out this line to make test case pass
    # sub: TestSubModule

    def __init__(self):
        super().__init__()
        self.sub = TestSubModule()

    def forward(self):
        y = MyTest()
        y.print()
        return self.sub()


m = TestModule()
print(m())

# pdb.set_trace()
scripted_m = torch.jit.script(m)
print(scripted_m.graph)
print(scripted_m())
