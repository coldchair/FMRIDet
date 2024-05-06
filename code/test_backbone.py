import torch
from projects.DETR_fmri.codetr.my_backbone import *
import sys
from projects.DETR_fmri.codetr.my_base_modules import TransformerPredictor, VisionTransformer_3D

from projects.DETR_fmri.codetr.my_backbone_3d_resnet import *
from projects.DETR_fmri.codetr.my_backbone_vit3d import Backbone_vit3d

def get_module_memory_size(module):
    size = sys.getsizeof(module)
    for param in module.parameters():
        size += param.data.element_size() * param.data.nelement()
    for buffer in module.buffers():
        size += buffer.element_size() * buffer.nelement()
    return size

import torch
from torch import nn


if __name__ == '__main__':
    bs = 8
    c = 256
    n = 25
    m = 25
    fmri_len = 26688

    n = 81
    m = 104
    h = 83

    n = 42
    m = 46
    h = 61

    inputs = torch.rand(bs, n, m, h)
    model = Backbone_vit3d(input_size=(n, m, h), out_channels=64)
    #统计 model 的参数量
    print(get_module_memory_size(model))
    outputs = model(inputs)[0]
    print(outputs.shape)

    # inputs = torch.rand(bs, 1, n, m, h)
    # model = VisionTransformer_3D(patch_size=8, width=256, layers=6, heads=8,
    #                              input_resolution=(n, m, h), num_class_embeddings=25)
    # outputs = model(inputs)
    # print(outputs.shape)

    # inputs = torch.rand(bs, n, m, h)
    # model = Backbone_3d()
    # outputs = model(inputs)[0]
    # print(outputs.shape)

    # inputs = torch.rand(bs, 1, n, m, h)
    # model = ResNet3D(BasicBlock3D, [2, 2, 2, 2])
    # model2 = ChannelMapper3D(in_channels = 256, out_channels=64)
    # outputs = model(inputs)
    # outputs = model2(outputs)
    # print(outputs.shape)


    # fmri_len = 26880

    # inputs = torch.rand(bs, fmri_len)
    # model = Backbone_fmri_seperate()
    # outputs = model(inputs)[0]
    # print(outputs.shape)

    # inputs = torch.rand(bs, 1, fmri_len, dtype = torch.float64)
    # model = Backbone_Conv()
    # outputs = model(inputs)[0]
    # print(outputs.shape)

    # inputs = torch.rand(bs, 1, fmri_len)
    # model = nn.Conv1d(kernel_size=32, stride=16, in_channels=1, out_channels=256)
    # outputs = model(inputs)
    # print(outputs.shape)

    # inputs = torch.rand(bs, fmri_len)
    # model = Backbone_fmri_transformer()
    # outputs = model(inputs)[0]
    # print(outputs.shape)

    # inputs = torch.rand(bs, 100, c)
    # model = TransformerPredictor(100, c, 6, 8, 2048, 4096, max_len = 100)
    # print(get_module_memory_size(model))
    # outputs = model(inputs)
    # print(outputs.shape)

    # # 创建形状为 bs * n * m 的向量
    # vector1 = torch.randn(bs, n, m).unsqueeze(1)

    # # 创建形状为 bs * c * n * m 的向量
    # vector2 = torch.randn(bs, c, n, m)

    # # 执行点乘操作
    # result = torch.mean(vector1 * vector2)

    # print(result)

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