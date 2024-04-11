from mmdet.apis import DetInferencer

# inferencer = DetInferencer(model='co_dino_5scale_swin_l_16xb1_16e_o365tococo.py',
#                            weights='co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth',
#                            device='cuda:2')

inferencer = DetInferencer('detr_r50_8xb2-150e_coco', device = 'cuda:0')

img_path = '/home/hdeng/diffusion/StableDiffusionReconstruction-main/all_images/images/'

img_list = []
n = 100;
for i in range(n):
    img_list.append(img_path + str(i).zfill(6) + '.png')

# 推理示例图片
inferencer(img_list,
           out_dir='outputs/',
           no_save_pred=False,
           batch_size=4,) 

