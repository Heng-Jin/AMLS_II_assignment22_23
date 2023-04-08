import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging
import torchvision.transforms.functional as f

# Recommended normalization params
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_augmentation = False  # data augmentation is leveraged if True

class bbox_padding:
    '''
    padding images into square
    '''
    def __init__(self, img_size):
        '''

        Args:
            img_size: image size to be padded to
        '''
        self.img_size = img_size

    def __call__(self, image: Image) -> torch.Tensor:
        """padding image with zero to a fixed width/height ratio

        Args:
            image (torch.Tensor): image
            ratio (float): ratio

        Returns:
            Union[torch.Tensor, Image]:
        """

        # assert ratio > 0, ratio
        if torch.is_tensor(image):
            src_h, src_w = image.shape[-2:]
        else:
            src_w, src_h = image.size
        padding_w = 0
        padding_h = 0
        if src_w < src_h:
            padding_w = int((src_h - src_w) // 2)
        else:
            padding_h = int((src_w - src_h) // 2)
        return transforms.functional.pad(image, padding=(padding_w, padding_h), padding_mode='symmetric')

# class extract_channel:
#     def __init__(self, channel):
#         self.channel = channel
#
#     def __call__(self, image: Image) -> torch.Tensor:
#         if self.channel == 'y':

# data augmentation setting

transforms_hr = transforms.Compose([
                    bbox_padding([510, 510]),
                    transforms.Resize([510, 510]),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=mean, std=std),
])

transforms_lr = transforms.Compose([
                    bbox_padding([255, 255]),
                    transforms.Resize([255, 255]),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=mean, std=std),
])

class amls_dataset(Dataset):
    '''
    Dataset for amlsII, this is for model of Y channel input only, Cb Cr will be discarded
    '''
    def __init__(self, path, mode):
        '''

        Args:
            path: data save path
            mode: "training", "test", or "inference"
        '''
        self.mode = mode
        if data_augmentation is True and self.mode == 'training':
            print('--- data agumentation ---')

        x_path_folder = path / 'X8'
        y_path_folder = path / 'X4'

        x_paths = list(x_path_folder.glob('*.png'))
        y_paths = list(y_path_folder.glob('*.png'))

        if len(x_paths) != len(y_paths):
            logging.warning('-----label amount dismatch with img amount-----')
            print('-----LR amount Mismatch with HR amount-----')
            print(len(x_paths), len(y_paths))

        x_paths_dict = {}
        y_paths_dict = {}
        for x_path in x_paths:
            key = str(x_path.stem)[:-2]
            x_paths_dict[key] = x_path
        for y_path in y_paths:
            key = str(y_path.stem)[:-2]
            y_paths_dict[key] = y_path

        self.dataset_list = list()  # format: [ind, lr_img_path, hr_img_path, lr_size, hr_size]
        for key, x_path in x_paths_dict.items():
            y_path = y_paths_dict[key]
            # ind_x = int(str(x_path.stem)[:-2])
            # ind_y = int(str(y_path.stem)[:-2])

            with Image.open(x_path) as img1, Image.open(y_path) as img2:
                width1, height1 = img1.size
                width2, height2 = img2.size
                if width1 == width2 // 2 and height1 == height2 // 2:
                    temp = [key, x_path, y_path, [width1, height1], [width2, height2]]
                    self.dataset_list.append(temp)
                else:
                    logging.error('-----LR / HR image size Mismatch-----')
                    print(f'{x_path} is not satisfy the 2X relationship with {y_path}', width1, height1, width2 / 2, height2 / 2)

    def img_aug(self, lr_image, hr_image):
        '''
        image data augmentation to make sure the lr images and hr images are transformed by identical operation
        Args:
            lr_image: LR image
            hr_image: HR image

        Returns: LR image and Hr image after transformation

        '''
        angle = random.randint(-180, 180)
        h = random.randint(0, 1)
        v = random.randint(0, 1)
        # lr_image = transforms.functional.rotate(lr_image, angle, interpolation=f.InterpolationMode.BICUBIC, expand=False)
        # hr_image = transforms.functional.rotate(hr_image, angle, interpolation=f.InterpolationMode.BICUBIC, expand=False)
        if h==1:
            lr_image = transforms.functional.hflip(lr_image)
            hr_image = transforms.functional.hflip(hr_image)

        if v==1:
            lr_image = transforms.functional.vflip(lr_image)
            hr_image = transforms.functional.vflip(hr_image)
        return lr_image, hr_image

    def __getitem__(self, index):
        '''

        Args:
            index: number index of data

        Returns: data

        '''
        img_lr = Image.open(self.dataset_list[index][1])
        img_hr = Image.open(self.dataset_list[index][2])

        #fix the even size problem
        # width, height = self.dataset_list[index][4]
        # img_hr = img_hr.resize((width//2*2, height//2*2), Image.BICUBIC)

        img_lr = img_lr.convert("YCbCr")
        img_hr = img_hr.convert("YCbCr")
        img_lr, cb_lr, cr_lr = img_lr.split()
        img_hr, _, _ = img_hr.split()

        if data_augmentation is True and self.mode == 'training':
            img_lr, img_hr = self.img_aug(img_lr, img_hr)

        img_lr = transforms_lr(img_lr)
        img_hr = transforms_hr(img_hr)

        return img_lr, img_hr

    def __len__(self):
        return len(self.dataset_list)



