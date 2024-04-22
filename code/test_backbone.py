import torch
from projects.DETR_fmri.codetr.my_backbone import Backbone_fmri_resnet1d_2_imgfeat
import sys

def get_module_memory_size(module):
    size = sys.getsizeof(module)
    for param in module.parameters():
        size += param.data.element_size() * param.data.nelement()
    for buffer in module.buffers():
        size += buffer.element_size() * buffer.nelement()
    return size

import torch

if __name__ == '__main__':
    bs = 8
    c = 256
    n = 25
    m = 25

    # 创建形状为 bs * n * m 的向量
    vector1 = torch.randn(bs, n, m).unsqueeze(1)

    # 创建形状为 bs * c * n * m 的向量
    vector2 = torch.randn(bs, c, n, m)

    # 执行点乘操作
    result = torch.mean(vector1 * vector2)

    print(result)

    # in_channels = 1
    # x = torch.randn(size=(8,1,26688))
    # model = Backbone_fmri_resnet1d_2_imgfeat()
    # print(get_module_memory_size(model))
    # output = model(x)[0]
    # # 计算损失函数
    # target_tensor = torch.randn(size=(8, 256, 25, 25))
    # loss = torch.nn.functional.mse_loss(output, target_tensor)
    # # 反向传播
    # loss.backward()
    # print(get_module_memory_size(model))
    # print(output.shape)