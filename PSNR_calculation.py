import torch
import pathlib
import numpy as np
import math
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image

origin_img = True
RGB = False

x_path_folder = pathlib.Path.cwd().parent / 'data' / 'DIV2K_valid_LR_bicubic' / 'X8'
# x_path_folder = pathlib.Path(r'D:\UCL_codes\0135\assignment\test\RGB_FSRCNN_train_test_04_03_14_37\test_image_inference')
# x_path_folder = pathlib.Path(r'D:\UCL_codes\0135\assignment\test\FSRCNN_train_test_04_01_23_54\test_image_inference')
# x_path_folder = pathlib.Path(r'D:\UCL_codes\0135\assignment\test\GAN_FSRCNN_train_test_04_03_14_23\test_image_inference')
# x_path_folder = pathlib.Path(r'D:\UCL_codes\0135\assignment\test\FSRCNN_train_test_04_02_22_13_residual\test_image_inference')
# x_path_folder = pathlib.Path(r'D:\UCL_codes\0135\assignment\test\test_image_inference_pretrain_model')
# x_path_folder = pathlib.Path(r'D:\UCL_codes\0135\assignment\test\FSRCNN_train_test_04_02_17_59_pretrain\test_image_inference')
y_path_folder = pathlib.Path.cwd().parent / 'data' / 'DIV2K_valid_LR_bicubic' / 'X4'


class bbox_padding:
    def __init__(self, img_size):
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

def psnr(img1, img2):
    mse = torch.mean(torch.square(img1 - img2), axis=(0,1,2))
    if torch.any(mse == 0):  # if mse==0 in any image
        return torch.tensor(float('inf'))
    max_pixel = 1.0  # based on assumption that the max value of pixel is 1
    psnr = 20 * torch.log10((max_pixel) / torch.sqrt(mse))
    return psnr

def calculate_psnr(image1, image2):
    """
    计算两个PIL.Image格式的图片的PSNR
    :param image1: 第一个图片，PIL.Image格式
    :param image2: 第二个图片，PIL.Image格式
    :return: 两个图片的PSNR值
    """
    # 将图片转换为numpy数组
    img1 = np.array(image1)
    img2 = np.array(image2)

    # 计算均方误差（MSE）
    mse = np.mean((img1 - img2) ** 2)

    # 计算峰值信噪比（PSNR）
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr

transforms_lr = transforms.Compose([
                    bbox_padding([255, 255]),
                    transforms.Resize([255, 255]),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=mean, std=std),
])

transforms_hr = transforms.Compose([
                    bbox_padding([510, 510]),
                    transforms.Resize([510, 510]),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=mean, std=std),
])



x_paths = list(x_path_folder.glob('*.png'))
y_paths = list(y_path_folder.glob('*.png'))

if len(x_paths) != len(y_paths):
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

test_psnr_sum, psnr_temp, n = 0.0, 0.0, 0

for key, x_path in x_paths_dict.items():
    lr_img = Image.open(x_path)
    if RGB == False:
        lr_img = lr_img.convert("YCbCr")
        lr_img, cb_lr, cr_lr = lr_img.split()

    if origin_img == True:
        lr_img = transforms_lr(lr_img)
        lr_img = transforms.Resize([510, 510])(lr_img)
    else:
        lr_img = transforms.ToTensor()(lr_img)

    y_path = y_paths_dict[key]
    hr_img = Image.open(y_path)
    if RGB == False:
        hr_img = hr_img.convert("YCbCr")
        hr_img, cb_hr, cr_hr = hr_img.split()

    hr_img = transforms_hr(hr_img)

    psnr_temp = psnr(hr_img, lr_img).cpu().item()
    # psnr_temp = calculate_psnr(hr_img, lr_img)
    test_psnr_sum += psnr_temp
    n += 1

    print(n, psnr_temp)

print('---the psnr_mean of lr images (resized to hr sizes) in validation dataset is  %.4f ---', test_psnr_sum/n)




