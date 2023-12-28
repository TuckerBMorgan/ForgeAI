import torch

a = torch.ones((2, 2), requires_grad=True)
b = torch.ones((2, 2), requires_grad=True)
c = a + b
e = -(c.log().mean())
e.backward()

print(a.grad)