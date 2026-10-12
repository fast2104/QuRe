import json
import numpy as np
import os
from typing import List
from pathlib import Path

from torch.utils.data import Dataset
from data.utils import targetpad_transform, _get_img_from_path
from data.abc import AbstractBaseDataset

import PIL
import PIL.Image


from pathlib import Path
base_path = Path("/home/iec/sontung_veryIMPORTANT/QuRe/fashioniq")
target_ratio = 1.25

class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split, dress_types, mode, preprocess: callable, config):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = targetpad_transform(target_ratio, config['img_size'])
        # self.preprocess = image_transform(config, split)
        self.mode = mode
        self.split = split
        self.config = config
        self.dress_types = dress_types

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(base_path /  'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']
                rel_caption = image_captions[0].strip('.?, ').capitalize() + " and " + image_captions[1].strip('.?, ')
                # rel_caption = caption_post_process(rel_caption)

                if self.split == 'train':
                    reference_image_path = base_path / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    # reference_image = _get_img_from_path(reference_image_path, self.preprocess)
                    target_name = self.triplets[index]['target']
                    target_image_path = base_path / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    # target_image = _get_img_from_path(target_image_path, self.preprocess)

                    return reference_image, target_image, target_name, rel_caption

                elif self.split == 'val':
                    reference_image_path = base_path / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    # reference_image = _get_img_from_path(reference_image_path, self.preprocess)
                    target_name = self.triplets[index]['target']
                    return reference_image, rel_caption, target_name

                elif self.split == 'test':
                    reference_image_path = base_path / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, rel_caption

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                # image = _get_img_from_path(image_path, self.preprocess)
                return image, image_name

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    @classmethod
    def code(cls):
        return 'fashionIQ'

    @classmethod
    def all_codes(cls):
        return ['fashionIQ']
    
    @classmethod
    def all_subset_codes(cls):
        return ['dress', 'shirt', 'toptee']

    @classmethod
    def vocab_path(cls):
        return None
        
##############################################################
############## For User Suvery Generation ####################
##############################################################

def _get_img_caption_json(dataset_root, clothing_type, split):
    with open(os.path.join(dataset_root, 'captions', 'cap.{}.{}.json'.format(clothing_type, split))) as json_file:
        img_caption_data = json.load(json_file)
    return img_caption_data

def _get_img_caption_txt(dataset_root, clothing_type, split, config):
    with open(os.path.join(dataset_root, 'captions_pairs', 'fashion_iq-{}-cap-{}.txt'.format(split, clothing_type))) as f:
        file_content = f.readlines()
    return file_content

def _get_img_split_json_as_list(dataset_root, clothing_type, split):
    with open(os.path.join(dataset_root, 'image_splits', 'split.{}.{}.json'.format(clothing_type, split))) as json_file:
        img_split_list = json.load(json_file)
    return img_split_list

def _create_img_path_from_id(root, id):
    return os.path.join(root, '{}.jpg'.format(id))

def _get_img_path_using_idx(img_caption_data, img_root, idx, is_ref=True):
    img_caption_pair = img_caption_data[idx]
    key = 'candidate' if is_ref else 'target'

    img = _create_img_path_from_id(img_root, img_caption_pair[key])
    id = img_caption_pair[key]
    return img, id

def _get_modifier(img_caption_data, idx, reverse=False):
    img_caption_pair = img_caption_data[idx]
    cap1, cap2 = img_caption_pair['captions']
    return _create_modifier_from_attributes(cap1, cap2) if not reverse else _create_modifier_from_attributes(cap2, cap1)

def _cat_captions(caps):
    I = []
    for i in range(len(caps)):
        if i % 2 == 0:
            I.append(_create_modifier_from_attributes(caps[i], caps[i+1]))
        else:
            I.append(_create_modifier_from_attributes(caps[i], caps[i-1]))
    return I

def _create_modifier_from_attributes(ref_attribute, targ_attribute):
    return ref_attribute + " and " + targ_attribute

def caption_post_process(s):
    return s.strip().replace('.', 'dotmark').replace('?', 'questionmark').replace('&', 'andmark').replace('*', 'starmark')

class AbstractBaseFashionIQDataset(AbstractBaseDataset):

    @classmethod
    def code(cls):
        return 'fashionIQ'

    @classmethod
    def all_codes(cls):
        return ['fashionIQ']
    
    @classmethod
    def all_subset_codes(cls):
        return ['dress', 'shirt', 'toptee']
    
    @classmethod
    def vocab_path(cls):
        return _DEFAULT_FASHION_IQ_VOCAB_PATH
    

class FashionIQUserSurveyDataset(AbstractBaseFashionIQDataset):
    """
    Fashion200K dataset.
    Image pairs in {root_path}/image_pairs/{split}_pairs.pkl

    """

    def __init__(self, root_path=base_path, clothing_type='dress', split='train',
                 img_transform=None, id_transform=None, config=None):
        super().__init__(root_path, split, img_transform)
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.clothing_type = clothing_type
        self.split = split
        self.img_transform = img_transform
        self.id_transform = id_transform
        self.img_caption_data = _get_img_caption_txt(root_path, clothing_type, split, config)
        self.ref_img_path = np.array([ff.strip().split(';')[0] for ff in self.img_caption_data])
        self.targ_img_path = np.array([ff.strip().split(';')[1] for ff in self.img_caption_data])
        self.caps = [ff.strip('\n').split(';')[-1] for ff in self.img_caption_data]
        self.caps_cat = _cat_captions(self.caps)
        self.config = config


    def __getitem__(self, idx):

        ref_img_path = os.path.join(self.img_root_path, self.ref_img_path[idx])
        targ_img_path = os.path.join(self.img_root_path, self.targ_img_path[idx])
        reference_img = _get_img_from_path(ref_img_path, self.img_transform)
        target_img = _get_img_from_path(targ_img_path, self.img_transform)


        sentences = self.caps_cat[idx]
        sentences = caption_post_process(sentences)
        # modifier = self.text_transform(sentences) if self.text_transform else sentences

        ref_id = self.ref_img_path[idx].split('/')[-1].split('.')[0]
        targ_id = self.targ_img_path[idx].split('/')[-1].split('.')[0]

        if self.id_transform:
            ref_id = self.id_transform(ref_id)
            targ_id = self.id_transform(targ_id)


        return reference_img, ref_img_path, target_img, targ_img_path, targ_id, sentences

    def __len__(self):
        return len(self.img_caption_data)# * 2


class FashionIQUserSurveyTestSampleDataset(AbstractBaseFashionIQDataset):
    """
    FashionIQ Test (Samples) dataset.
    indexing returns target samples and their unique ID
    """

    def __init__(self, root_path=base_path, clothing_type='dress', split='val',
                 img_transform=None, config=None):
        super().__init__(root_path, split, img_transform)
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.clothing_type = clothing_type
        self.img_transform = img_transform

        #self.img_list = _get_img_split_json_as_list(root_path, clothing_type, split)

        ''' Uncomment below for VAL Evaluation method '''
        self.img_caption_data = _get_img_caption_txt(root_path, clothing_type, split, config)
        self.img_list = []
        for d in self.img_caption_data:
            ref = d.split(';')[0].split('/')[-1].split('.')[0] #dress/xxx.jpg --> xxx
            targ = d.split(';')[1].split('/')[-1].split('.')[0] #dress/xxx.jpg --> xxx
            self.img_list.append(ref)
            self.img_list.append(targ)
        self.img_list = list(set(self.img_list))

    def __getitem__(self, idx, use_transform=True):
        img_transform = self.img_transform if use_transform else None
        img_id = self.img_list[idx]
        img_path = _create_img_path_from_id(os.path.join(self.img_root_path,self.clothing_type), img_id)

        target_img = _get_img_from_path(img_path, img_transform)

        return target_img, img_path, img_id

    def __len__(self):
        return len(self.img_list)