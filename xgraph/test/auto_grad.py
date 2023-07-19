import torch
import copy
a = torch.rand((1, 1, 3, 10), requires_grad=True)
b = torch.rand((1, 1, 3, 3), requires_grad=True)
c = b @ a
d = copy.deepcopy(b) # necessarily use deepcopy
e = d @ c
w = torch.rand((10, 1), requires_grad=True)
o = e.mean(dim=2) @ w

grads = torch.autograd.grad(outputs=o.squeeze(), inputs=d, create_graph=True)
print(grads)
grad_d11 = grads[0][:, :, 1, 1]
r11 = grad_d11 * d[:, :, 1, 1]
print(r11)

grads2 = torch.autograd.grad(outputs=r11.squeeze(), inputs=b, create_graph=True)
print(grads2)
grad_b12 = grads2[0][:, :, 1, 2]
r211 = grad_b12 * b[:, :, 1, 2]
print(r211)