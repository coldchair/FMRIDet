import os

# zip_dir = '/home/hdeng/diffusion/fmri-detection/mmdetection/data/OpenDataLab___COCO_2017/raw'
zip_dir = '/mnt/workspace/maxinzhu/denghan/coco/OpenDataLab___COCO_2017/raw'
# save_dir = '/home/hdeng/diffusion/fmri-detection/mmdetection/data/coco'
save_dir = '/mnt/workspace/maxinzhu/denghan/coco'

name_list = ['Annotations', 'Images']

for name in name_list:
    zip_path = os.path.join(zip_dir, name)
    save_path = save_dir
    for file in os.listdir(zip_path):
        if (file.endswith('.zip')):
            command = f'unzip -o {os.path.join(zip_path, file)} -d {save_path}'
            print(command)
            os.system(command)
