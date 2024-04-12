import argparse
import os
import numpy as np
import pandas as pd
from nsd_access import NSDAccess
import scipy.io

from config import NSD_ROOT_DIR, DATA_ROOT_DIR
from tqdm import tqdm
from PIL import Image


nsda = NSDAccess(NSD_ROOT_DIR)
nsd_expdesign = scipy.io.loadmat(os.path.join(NSD_ROOT_DIR, 'nsddata/experiments/nsd/nsd_expdesign.mat'))

n = 73000

save_dir = os.path.join(DATA_ROOT_DIR, 'all_images')
os.makedirs(save_dir, exist_ok=True)

batch_size = 5000

for i in tqdm(range(0, n, batch_size)):
    j = min(i + batch_size, n)
    images = nsda.read_images(list(range(i, j)))
    for k in tqdm(range(i, j)):
        # save the image as .png, image is numpy array
        image = Image.fromarray(images[k - i])
        image_path = os.path.join(save_dir, f'{k:06}.png')
        image.save(image_path)