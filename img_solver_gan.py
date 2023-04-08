import pathlib
import argparse
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from dataset_gan import amls_dataset, bbox_padding
from model import FSRCNN

'''
This python script generate the images of test data by assigned model
New images will be saved in the model folder
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8

parser = argparse.ArgumentParser(description='AMLS assignment image solver')
parser.add_argument('--input', type=str, required=False, default=r'D:\UCL_codes\0135\assignment\data\DIV2K_valid_LR_bicubic', help='test image to use')
parser.add_argument('--model', type=str, default=r'D:\UCL_codes\0135\assignment\test\GAN_FSRCNN_train_test_04_05_18_38\_epoch_105_lr_tensor(0.0002)_04_05_20_46_35.pth', help='model file to use')
# parser.add_argument('--model', type=str, default=r'D:\UCL_codes\0135\assignment\test\RGB_FSRCNN_train_test_04_03_22_30\_epoch_298_lr_tensor(0.0005)_04_03_23_51_27.pth', help='model file to use')
# parser.add_argument('--model', type=str, default=r'D:\UCL_codes\0135\assignment\test\GAN_FSRCNN_train_test_04_03_14_23\_epoch_221_lr_tensor(0.0002)_04_03_18_58_08.pth', help='model file to use')
args = parser.parse_args()
print(args)

save_path = pathlib.Path(args.model).parent / 'test_image_inference'
save_path.mkdir(exist_ok=True)

test_dataset = amls_dataset(pathlib.Path(args.input), "inference")
dataset_list = test_dataset.dataset_list # format: [ind, lr_img_path, hr_img_path, lr_size, hr_size]
test_iter = DataLoader(test_dataset, batch_size=batch_size)

net = FSRCNN(num_channels=3)
net.load_state_dict(torch.load(pathlib.Path(args.model)))
net = net.to(device)

with torch.no_grad():
    net.eval()  # evaluate mode
    test_psnr_sum, n2 = 0.0, 0
    test_result_list = []
    for X, y in test_iter:
        y_hat = net(X.to(device)).clamp(0.0, 1.0).cpu()
        # print(y_hat.shape, cb_lr.shape)
        for i, y_hr in enumerate(y_hat):
            out_img = transforms.ToPILImage()(y_hr).convert('RGB')
            out_path = save_path / (pathlib.Path(dataset_list[n2][2]).name)
            out_img.save(out_path)
            n2 += 1
            print(n2)




