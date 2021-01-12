import torch

@torch.jit.interface
class TestInterface(torch.nn.Module):
    def __init__(self):
        pass

    def add(self, x: int) -> int:
        pass

class TestModule(torch.nn.Module):
    def __init__(self, v: int):
        super().__init__()
        self.x = v

    @torch.jit.export
    def add(self, x: int):
        return self.x + x

def fn(m: TestInterface):
    return m.add(1)

m = torch.jit.script(TestModule(2))
print(fn(m))
