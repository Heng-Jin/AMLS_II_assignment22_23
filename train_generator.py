import csv
import numpy as np
import logging
import pathlib
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from PIL import Image
from matplotlib import pyplot as plt
from dataset_gan import amls_dataset
from model import FSRCNN, FSRCNN_Residual

# Recommended normalization params
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# hyper parameters
lr = 0.0002
lr = torch.tensor(lr, requires_grad=False)
epoch_num = 300
batch_size = 64
pretrain = False
residual = False

Loss_list = []
Accuracy_train_list = []
psnr_test_list = []

# train_datapath = pathlib.Path(r'D:\UCL_codes\0135\data\DIV2K_train_LR_bicubic')
# test_datapath = pathlib.Path(r'D:\UCL_codes\0135\data\DIV2K_valid_LR_bicubic')

train_datapath = pathlib.Path.cwd().parent / 'data' / 'DIV2K_train_LR_bicubic'
test_datapath = pathlib.Path.cwd().parent / 'data' / 'DIV2K_valid_LR_bicubic'

pretrained_model_path = pathlib.Path.cwd() / 'fsrcnn_x2.pth'

parent_path = pathlib.Path.cwd()
model_save_path = parent_path / ("RGB_FSRCNN_train_test_" + str(time.strftime("%m_%d_%H_%M", time.localtime())))
model_save_path.mkdir(exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')

log_path = model_save_path / ("FSRCNN_train_test_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".log")
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=log_path,
                    filemode='a')

def create_csv(path, result_list):
    # save predict labels of test dataset
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["origin psnr", "model psnr"])
        csv_write.writerows([i] for i in result_list)

def plot_save(loss_list, acc_list):
    # plot temporary loss of training and accuracy of test dataset after each epoch training
    x1 = range(len(acc_list))
    x2 = range(len(loss_list))
    y1 = acc_list
    y2 = loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test MSE vs. epoches')
    plt.ylabel('Test MSE')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Training loss vs. iteration')
    plt.ylabel('Training loss')
    # plt.show()
    plt.savefig(model_save_path / ("epoch_" + str(epoch_num) + "_lr_" + str(lr) + "_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) +".jpg"))
    create_csv(model_save_path / 'acc_list.csv', acc_list)
    create_csv(model_save_path / 'loss_list.csv', loss_list)

def psnr(img1, img2):
    mse = torch.mean(torch.square(img1 - img2), axis=(1,2,3))
    if torch.any(mse == 0):  # if mse==0 in any image
        return float('inf')
    max_pixel = 1.0  # based on assumption that the max value of pixel is 1
    psnr = 10 * torch.log10((max_pixel ** 2) / torch.sqrt(mse))
    return psnr

def train(net, train_iter, test_iter, criterion, optimizer, num_epochs):
    net = net.to(device)
    logging.info("-----training on %s-----", str(device))
    print("-----training on ", str(device), "-----")
    print(net)
    whole_batch_count = 0
    # training loop
    for epoch in range(num_epochs):
        start = time.time()
        net.train()  # trainning mode
        train_loss_sum, n, batch_count = 0.0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            # y = y.to(torch.long)
            optimizer.zero_grad()
            y_hat = net(X)
            # print(y_hat.type(),y.type())
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            n += y.shape[0]
            whole_batch_count += 1
            batch_count += 1
            temp_loss = train_loss_sum / whole_batch_count
            Loss_list.append(loss.item())
            logging.info('-epoch %d, batch_count %d, img nums %d, loss temp %.4f, time %.1f sec,'
                  % (epoch + 1, whole_batch_count, n, loss.item(), time.time() - start))
            print('-epoch %d, batch_count %d, img nums %d, loss temp %.4f, time %.1f sec'
                  % (epoch + 1, whole_batch_count, n, loss.item(), time.time() - start))

        # test dataset inference will be done after each epoch
        with torch.no_grad():
            net.eval()  # evaluate mode
            test_psnr_sum, lr_psnr_sum, n2 = 0.0, 0.0, 0
            test_result_list=[]
            for X, y in test_iter:
                y_hat = net(X.to(device)).clamp(0.0, 1.0)
                y = y.to(device)
                psnr_temp = psnr(y, y_hat).cpu()
                # x_resized = transforms.Resize([510, 510])(x)
                # lr_psnr_temp = psnr(y, x_resized).cpu()
                # print(type(psnr_temp),psnr_temp.shape)
                # temp = torch.stack((y_hat.argmax(dim=1).int(), y.to(device).int(), y_hat.argmax(dim=1) == y.to(device)), 1).tolist()
                # print(psnr_temp.float().tolist())
                test_result_list.extend(psnr_temp.float().tolist())
                # test_result_list.extend(torch.stack((lr_psnr_temp, psnr_temp), 1).tolist())
                n2 += y.shape[0]
                test_psnr_sum += psnr_temp.sum().item()
                # lr_psnr_sum += lr_psnr_temp.sum().item()
                temp_psnr_test = test_psnr_sum / n2
                logging.info('---epoch %d, img nums %d, psnr_mean %.4f, loss %.4f, time %.1f sec---'
                      % (epoch + 1, n, temp_psnr_test, temp_loss, time.time() - start))
                print('---epoch %d, img nums %d, psnr_mean %.4f, loss %.4f, time %.1f sec---'
                      % (epoch + 1, n, temp_psnr_test, temp_loss, time.time() - start))

        psnr_test_list.append(temp_psnr_test)

        result_path = model_save_path / ("epoch_" + str(epoch) + "_lr_" + str(lr) +"_test_result.csv")
        create_csv(result_path, test_result_list)

        torch.save(net.state_dict(),
                   model_save_path / ("_epoch_" + str(epoch) + "_lr_" + str(lr) + "_" + str(
                       time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".pth"))

def run():
    # main function described in report
    train_dataset = amls_dataset(train_datapath, "training")
    test_dataset = amls_dataset(test_datapath, "test")

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    if residual == True:
        net = FSRCNN_Residual(num_channels=3)
    else:
        net = FSRCNN(num_channels=3)

    if pretrain == True and residual == False:
        net.load_state_dict(torch.load(pretrained_model_path))

    optimizer = optim.Adam([
        {'params': net.first_part.parameters()},
        {'params': net.mid_part.parameters()},
        {'params': net.last_part.parameters(), 'lr': lr * 0.1}
    ], lr=lr)

    loss = torch.nn.MSELoss()
    train(net, train_iter, test_iter, loss, optimizer, num_epochs=epoch_num)

    plot_save(Loss_list, psnr_test_list)

    # torch.save(net.state_dict(),
    #            model_save_path / (task + "_epoch_" + str(epoch) + "_lr_" + str(lr) + "_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) +".pth"))

if __name__ == '__main__':
    run()
