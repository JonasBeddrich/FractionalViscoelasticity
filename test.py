import torch 
x = torch.randn(2, 3)
x
torch.cat((x, x, x), 0)

print(torch.cat(([x, x])))