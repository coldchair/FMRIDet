import openxlab

ak = 'ddlwzyzjvqkree6y9z1x'
sk = 'ygxr8jrdlbzykwvl1rzn6vayx0m2pmz9d3kjpxoq'

openxlab.login(ak=ak, sk=sk, relogin=True) #进行登录，输入对应的AK/SK

from openxlab.dataset import info
info(dataset_repo='OpenDataLab/COCO_2017') #数据集信息查看

from openxlab.dataset import query
# query(dataset_repo='OpenDataLab/COCO_2017') #数据集文件列表查看

from openxlab.dataset import get
# get(dataset_repo='OpenDataLab/COCO_2017', target_path='/home/hdeng/diffusion/fmri-detection/mmdetection/data')  # 数据集下载

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/COCO_2017',
         source_path='/raw',
         target_path='/mnt/workspace/maxinzhu/denghan/coco') #数据集文件下载