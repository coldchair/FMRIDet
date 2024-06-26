# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.coco import CocoDataset
import numpy as np
import os

@DATASETS.register_module()
class CocoNSDDataset(CocoDataset):
    """Dataset for NSD."""

    ANN_ID_UNIQUE = False

    def __init__(self,
                 index_file,
                 fmri_files_path,
                 input_type = 'one', # 'one' or 'multi' or '3d' or 'multi_single'
                 input_size = None, # None or list of int
                 input_dim = 1,
                 padding_zeros = 0,
                 fmri_suffix_name = '_tr', # '_tr' or '_ave_te'
                 fmri_prefix_name = None,
                 *args, **kwargs):
        self.index_file = index_file
        self.fmri_files_path = fmri_files_path
        self.input_dim = input_dim
        self.padding_zeros = padding_zeros
        self.input_type = input_type 
        self.input_size = input_size
        self.fmri_suffix_name = fmri_suffix_name
        self.fmri_prefix_name = fmri_prefix_name
        super().__init__(*args, **kwargs)
    
    def read_and_stack_fmri(self, fmri_files_path):
        if (self.input_type == 'one'):
            if (self.input_dim == 1):
                X = []
                for fmri_file in fmri_files_path:
                    cX = np.load(fmri_file).astype("float32")
                    X.append(cX)
                    if (self.padding_zeros > 0):
                        zeros = np.zeros((cX.shape[0], self.padding_zeros), dtype="float32")
                        X.append(zeros)
                X = np.hstack(X) # shape : (n_samples, n_voxels)
            else:
                X = []
                for fmri_file in fmri_files_path:
                    cX = np.load(fmri_file).astype("float32")
                    n = cX.shape[-1]
                    # print(cX.shape)
                    if (n % self.input_dim != 0):
                        cX = np.pad(cX, [(0, 0), (0, self.input_dim - n % self.input_dim)])
                    len = n // self.input_dim + (n % self.input_dim != 0)
                    cX = cX.reshape(-1, len, self.input_dim)
                    # print(cX.shape)
                    X.append(cX)
                X = np.concatenate(X, axis = 1) # shape : (n_samples, length, input_dim)

            self.fmri = X
            print("fmri shape : ", X.shape)
            return X
        else:
            ans = np.zeros((0, np.sum(self.input_size)), dtype="float32")
            for i, subj in enumerate(fmri_files_path):
                print(f'loading fmri data for subject {i}-th')
                if (subj != []):
                    X = []
                    for fmri_file in subj:
                        cX = np.load(fmri_file).astype("float32")
                        X.append(cX)
                        if (self.padding_zeros > 0):
                            zeros = np.zeros((cX.shape[0], self.padding_zeros), dtype="float32")
                            X.append(zeros)
                    X = np.hstack(X) # shape : (n_samples, n_voxels)
                else:
                    X = np.zeros((0, self.input_size[i]), dtype="float32")
                z0 = np.sum(self.input_size[:i]).astype("int")
                z1 = np.sum(self.input_size[i+1:]).astype("int")
                X = np.pad(X, [(0, 0), (z0, z1)])
                ans = np.concatenate([ans, X], axis = 0)
            print("fmri shape : ", ans.shape)
            self.fmri = ans
            return ans
                    


    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        # ------------------------------- modification ------------------------------- #
        # img_ids = self.coco.get_img_ids()
        if (self.input_type == 'one'):
            img_ids = np.load(self.index_file).tolist()
            self.read_and_stack_fmri(self.fmri_files_path)
        elif (self.input_type == 'multi'):
            img_ids = []
            for subj in self.index_file:
                if (subj != []):
                    img_ids_ = np.load(subj).tolist()
                    img_ids = img_ids + img_ids_
            self.read_and_stack_fmri(self.fmri_files_path)
        elif (self.input_type == '3d'):
            img_ids = np.load(self.index_file).tolist()
            self.fmri_files_path = [os.path.join(self.fmri_prefix_name,
                                                 f'{i:06}{self.fmri_suffix_name}.npy') for i in range(len(img_ids))]
        elif (self.input_type == 'multi_single'):
            img_ids = []
            self.fmri_files_path = []
            self.fmri_pd = []
            for i, subj in enumerate(self.index_file):
                if (subj != []):
                    img_ids_ = np.load(subj).tolist()
                    img_ids = img_ids + img_ids_

                    pf = [
                        os.path.join(self.fmri_prefix_name[i],
                                     f'{self.fmri_suffix_name}{j:06}.npy')
                        for j in range(len(img_ids_))]
                    self.fmri_files_path = self.fmri_files_path + pf

                    p = (np.sum(self.input_size[:i]).astype('int'), 
                         np.sum(self.input_size[i+1:]).astype('int'))
                    pd = [p for j in range(len(img_ids_))]
                    self.fmri_pd = self.fmri_pd + pd

                    print(f'i : {i}, len : {len(img_ids_)}')
            
        print(f'img_ids : {len(img_ids)}')

        if not hasattr(self, 'fmri_files_path'):
            self.fmri_files_path  = [None for i in range(len(img_ids))]
        
        if not hasattr(self, 'fmri'):
            self.fmri = [None for i in range(len(img_ids))]
        
        if not hasattr(self, 'fmri_pd'):
            self.fmri_pd = [None for i in range(len(img_ids))]

        # ------------------------------- modification ------------------------------- #

        data_list = []
        total_ann_ids = []
        for i, img_id in enumerate(img_ids):
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info,
                # ------------------------------- modification ------------------------------- #
                'fmri' :
                self.fmri[i],
                'fmri_path' : 
                self.fmri_files_path[i],
                'padding_zeros' :
                self.fmri_pd[i],
                # ------------------------------- modification ------------------------------- #
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # ------------------------------- modification ------------------------------- #
        data_info['fmri'] = raw_data_info['fmri']
        data_info['fmri_path'] = raw_data_info['fmri_path']
        data_info['padding_zeros'] = raw_data_info['padding_zeros']
        # ------------------------------- modification ------------------------------- #

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

