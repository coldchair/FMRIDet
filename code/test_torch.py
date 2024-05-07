import torch

# 创建一个三维tensor
a = torch.tensor([[[-1, 2], [3, -4]], [[-5, 6], [7, -8]]])

# 创建一个四维tensor
b = torch.randn(2, 2, 2, 3)

# 使用a<0生成一个布尔类型的tensor
mask = a < 0
print(mask)

# 使用生成的mask来索引b的前三维码
result = b[mask]

print(result)
print(result.size())
