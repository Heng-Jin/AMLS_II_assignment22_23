import csv
import numpy as np
import logging
import pathlib
import argparse
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from dataset import amls_dataset, bbox_padding
from model import FSRCNN

'''
This python script generate the images of test data by assigned model
New images will be saved in the model folder
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8

parser = argparse.ArgumentParser(description='AMLS assignment image solver')
parser.add_argument('--input', type=str, required=False, default=r'D:\UCL_codes\0135\assignment\data\DIV2K_valid_LR_bicubic', help='test image to use')
# parser.add_argument('--model', type=str, default=r'D:\UCL_codes\0135\assignment\test\FSRCNN_train_test_04_01_23_54\_epoch_297_lr_tensor(0.0002)_04_02_00_52_01.pth', help='model file to use')
# parser.add_argument('--model', type=str, default=r'D:\UCL_codes\0135\assignment\test\FSRCNN_train_test_04_02_22_13_residual\_epoch_299_lr_tensor(0.0002)_04_02_23_09_47.pth', help='model file to use')
# parser.add_argument('--model', type=str, default=r'D:\UCL_codes\0135\assignment\test\FSRCNN_train_test_04_02_17_59_pretrain\_epoch_292_lr_tensor(0.0002)_04_02_18_55_30.pth', help='model file to use')
parser.add_argument('--model', type=str, default=r'D:\UCL_codes\0135\assignment\test\fsrcnn_x2.pth', help='model file to use')

args = parser.parse_args()
print(args)

save_path = pathlib.Path(args.model).parent / 'test_image_inference_pretrain'
save_path.mkdir(exist_ok=True)

test_dataset = amls_dataset(pathlib.Path(args.input), "inference")
dataset_list = test_dataset.dataset_list # format: [ind, lr_img_path, hr_img_path, lr_size, hr_size]
test_iter = DataLoader(test_dataset, batch_size=batch_size)

net = FSRCNN()
net.load_state_dict(torch.load(pathlib.Path(args.model)))
net = net.to(device)

transforms_cbcr = transforms.Compose([
                    transforms.Resize([510, 510]),
                    transforms.ToPILImage(),
                    # transforms.Normalize(mean=mean, std=std),
])

with torch.no_grad():
    net.eval()  # evaluate mode
    test_psnr_sum, n2 = 0.0, 0
    test_result_list = []
    for X, y in test_iter:

        y_hat = net(X.to(device)).clamp(0.0, 1.0).cpu()
        cb_lr, cr_lr = torch.unbind(y, dim=1)
        # print(y_hat.shape, cb_lr.shape)
        for i, y_hr in enumerate(y_hat):
            y_hr = transforms.ToPILImage()(y_hr)
            cb_hr = transforms_cbcr(cb_lr[i])
            cr_hr = transforms_cbcr(cr_lr[i])
            out_img = Image.merge('YCbCr', [y_hr, cb_hr, cr_hr]).convert('RGB')
            out_path = save_path / (pathlib.Path(dataset_list[n2][2]).name)
            out_img.save(out_path)
            n2 += 1
            print(n2)




