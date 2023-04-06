import torchvision.transforms as transforms
from PIL import Image
import random
import random
import torchvision.transforms as transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as f


# 定义一个随机角度函数，用于生成随机角度



# 定义一个图像变换函数，用于对 LR 和 HR 图片进行随机旋转
class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        print(self.angle)
        return transforms.functional.rotate(img, self.angle)


# # 创建一个变换函数序列，用于对 LR 和 HR 图片进行随机旋转
# transform = transforms.Compose([
#     RandomRotation(random_angle()),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor()
# ])

# # 加载 LR 和 HR 图片，并对它们进行相同的随机旋转
# lr_image = Image.open('path/to/lr_image.png')
# hr_image = Image.open('path/to/hr_image.png')
# seed = random.randint(0, 2**32)  # 生成一个随机种子
# random.seed(seed)  # 设置随机种子
# lr_image = transform(lr_image)
# random.seed(seed)  # 使用相同的随机种子
# hr_image = transform(hr_image)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, lr_image_list, hr_image_list):
        self.lr_image_list = lr_image_list
        self.hr_image_list = hr_image_list

    def img_aug(self, lr_image, hr_image):
        angle = random.randint(-180, 180)
        h = random.randint(0, 1)
        v = random.randint(0, 1)
        lr_image = transforms.functional.rotate(lr_image, angle, interpolation=f.InterpolationMode.BICUBIC, expand=False)
        hr_image = transforms.functional.rotate(hr_image, angle, interpolation=f.InterpolationMode.BICUBIC, expand=False)
        if h==1:
            lr_image = transforms.functional.hflip(lr_image)
            hr_image = transforms.functional.hflip(hr_image)

        if v==1:
            lr_image = transforms.functional.vflip(lr_image)
            hr_image = transforms.functional.vflip(hr_image)
        return lr_image, hr_image

    def __getitem__(self, index):
        lr_image = Image.open(self.lr_image_list[index])
        hr_image = Image.open(self.hr_image_list[index])
        lr_image, hr_image = self.img_aug(lr_image, hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.lr_image_list)


lr_image_list = ['D:/UCL_codes/0135/assignment/data/DIV2K_train_LR_bicubic/X8/0002x8.png',
                 'D:/UCL_codes/0135/assignment/data/DIV2K_train_LR_bicubic/X8/0003x8.png',
                 'D:/UCL_codes/0135/assignment/data/DIV2K_train_LR_bicubic/X8/0004x8.png']
hr_image_list = ['D:/UCL_codes/0135/assignment/data/DIV2K_train_LR_bicubic/X4/0002x4.png',
                 'D:/UCL_codes/0135/assignment/data/DIV2K_train_LR_bicubic/X4/0003x4.png',
                 'D:/UCL_codes/0135/assignment/data/DIV2K_train_LR_bicubic/X4/0004x4.png']
d = MyDataset(lr_image_list, hr_image_list)
for i in range(10):
    lr, hr = d.__getitem__(i % 3)
    plt.imshow(lr)
    plt.show()
    plt.imshow(hr)
    plt.show()