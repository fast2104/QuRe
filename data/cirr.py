import json
import os

from data.utils import targetpad_transform
from torch.utils.data import Dataset

import PIL
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
from pathlib import Path
base_path = Path("/teamspace/studios/this_studio/fashioniq")
target_ratio = 1.25

class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable, config):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = targetpad_transform(target_ratio, config['img_size'])
        self.mode = mode
        self.split = split
        self.config = config

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(os.path.join(base_path , 'cirr' , 'captions' , f'cap.rc2.{split}.json')) as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(os.path.join(base_path , 'cirr' , 'image_splits' , f'split.rc2.{split}.json')) as f:
            self.name_to_relpath = json.load(f)

        self.image_names = list(self.name_to_relpath.keys())
        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = os.path.join(base_path  , self.name_to_relpath[reference_name])
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = os.path.join(base_path  , self.name_to_relpath[target_hard_name])
                    target_image = self.preprocess(PIL.Image.open(target_image_path))

                    return reference_image, target_image, target_hard_name, rel_caption

                elif self.split == 'val':
                    reference_image_path = os.path.join(base_path, self.name_to_relpath[reference_name])
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = os.path.join(base_path, self.name_to_relpath[target_hard_name])
                    target_image = self.preprocess(PIL.Image.open(target_image_path))

                    pair_id = self.triplets[index]['pairid']
                    return reference_name, reference_image, target_hard_name, target_image, rel_caption, pair_id, group_members
                    # target_hard_name = self.triplets[index]['target_hard']
                    # return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    reference_image_path = os.path.join(base_path, self.name_to_relpath[reference_name])
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))

                    pair_id = self.triplets[index]['pairid']
                    return reference_name, reference_image, rel_caption, pair_id, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = os.path.join(base_path, self.name_to_relpath[image_name])
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)

                # # This is only used for User survey generation
                # if self.split == 'val':
                #     return image, image_path, image_name
                return image, image_name

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
        
    @classmethod
    def code(cls):
        return 'cirr'

    @classmethod
    def all_codes(cls):
        return ['cirr']

    @classmethod
    def vocab_path(cls):
        return None

if __name__ == '__main__':

    # cirr_train_dataset = CIRRDataset('test1', 'classic', None, None)
    #
    # train_loader = DataLoader(dataset=cirr_train_dataset, batch_size=32,
    #                           pin_memory=False, collate_fn=collate_fn,
    #                           drop_last=True, shuffle=True)
    #
    # for idx, (reference_image, target_image, target_id, rel_caption) in enumerate(train_loader):
    #     print('hello')
    #
    # cirr_val_dataset = CIRRDataset('val', 'relative', None)
    # cirr_test1_dataset = CIRRDataset('test1', 'relative', None)
    #
    # print(len(cirr_train_dataset), len(cirr_val_dataset), len(cirr_test1_dataset))
    pass
