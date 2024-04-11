import os

zip_dir = '/home/hdeng/diffusion/fmri-detection/mmdetection/data/OpenDataLab___COCO_2017/raw'
save_dir = '/home/hdeng/diffusion/fmri-detection/mmdetection/data/coco'

name_list = ['Annotations', 'Images']

for name in name_list:
    zip_path = os.path.join(zip_dir, name)
    if (name == 'Annotations'):
        save_path = os.path.join(save_dir, 'annotations')
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = save_dir
    
    for file in os.listdir(zip_path):
        if (file.endswith('.zip')):
            command = f'unzip {os.path.join(zip_path, file)} -d {save_path}'
            print(command)
            os.system(command)
