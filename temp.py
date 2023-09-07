import torch
print(torch.__version__)

import piq


# loss = piq.HaarPSILoss()
# x = torch.rand(3, 3, 256, 256, requires_grad=True)
# y = torch.rand(3, 3, 256, 256)
# output = loss(x, y)
# print(output.item())

x = torch.randn(1,3, 3, 5)
print(x)
print(x[0].shape)
x = torch.permute(x[0],(2,1,0))
print(x[0])


