import csv
import numpy as np
import logging
import pathlib
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from dataset_gan import amls_dataset
from model import FSRCNN, FSRCNN_Residual
from torchvision.models.vgg import vgg19
from srgan import Discriminator, Generator

# Recommended normalization params
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# hyper parameters
lr_G = 0.0002
lr_G = torch.tensor(lr_G, requires_grad=False)
lr_D = 0.0002
lr_D = torch.tensor(lr_D, requires_grad=False)
momentum = 0.9
weight_decay = 0
epoch_num = 500
batch_size = 8
pretrain = False  # available only on FSRCNN
residual = False

Loss_list = []
Accuracy_train_list = []
psnr_test_list = []

pretrained_net_path = pathlib.Path.cwd() / 'RGB_FSRCNN_train_test_04_03_14_37'/ '_epoch_299_lr_tensor(0.0002)_04_03_16_00_30.pth'
pretrained_teacher_path = pathlib.Path.cwd() / 'DISCRIMINATOR_train_test_04_03_17_54'/ '_epoch_29_lr_tensor(0.0002)_04_03_18_10_11.pth'

train_datapath = pathlib.Path.cwd().parent / 'data' / 'DIV2K_train_LR_bicubic'
test_datapath = pathlib.Path.cwd().parent / 'data' / 'DIV2K_valid_LR_bicubic'

# pretrained_model_path = pathlib.Path.cwd() / 'fsrcnn_x2.pth'

parent_path = pathlib.Path.cwd()
model_save_path = parent_path / ("GAN_FSRCNN_train_test_" + str(time.strftime("%m_%d_%H_%M", time.localtime())))
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
    plt.savefig(model_save_path / ("epoch_" + str(epoch_num) + "_lr_" + str(lr_G) + "_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) +".jpg"))
    create_csv(model_save_path / 'acc_list.csv', acc_list)
    create_csv(model_save_path / 'loss_list.csv', loss_list)

def psnr(img1, img2):
    mse = torch.mean(torch.square(img1 - img2), axis=(1,2,3))
    if torch.any(mse == 0):  # if mse==0 in any image
        return float('inf')
    max_pixel = 1.0  # based on assumption that the max value of pixel is 1
    psnr = 10 * torch.log10((max_pixel ** 2) / torch.sqrt(mse))
    return psnr

def train(net, teacher, VGG_feature_model, train_iter, test_iter, G_criterion, D_criterion, G_optimizer, D_optimizer, num_epochs): #net, teacher, VGG_feature_model, train_iter, test_iter, loss, G_optimizer, D_optimizer, num_epochs=epoch_num
    net = net.to(device)
    teacher = teacher.to(device)
    VGG_feature_model = VGG_feature_model.to(device)
    logging.info("-----training on %s-----", str(device))
    print("-----training on ", str(device), "-----")
    print(net)
    whole_batch_count = 0
    # training loop
    for epoch in range(num_epochs):
        start = time.time()
        net.train()  # trainning mode
        teacher.train()
        G_train_loss_sum, D_train_loss_sum, n, batch_count = 0.0, 0.0, 0, 0
        for lr_images, hr_images in train_iter:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            temp_batch_size = lr_images.size()[0]
            y_real, y_fake = torch.ones(temp_batch_size).to(device), torch.zeros(temp_batch_size).to(device)
            # y = y.to(torch.long)

            # train Discriminator
            D_optimizer.zero_grad()
            # real class
            D_result = teacher(hr_images).squeeze()
            D_real_loss = D_criterion(D_result, y_real)
            D_real_loss.backward()
            # fake class
            G_result = net(lr_images)
            D_result = teacher(G_result).squeeze()
            D_fake_loss = D_criterion(D_result, y_fake)
            D_fake_loss.backward()
            # step
            D_optimizer.step()
            # loss calculation
            D_train_loss = D_real_loss + D_fake_loss

            # train Generator
            G_optimizer.zero_grad()
            # image loss
            G_result = net(lr_images)
            image_loss = G_criterion(G_result, hr_images)
            # adversarial loss
            D_result = teacher(G_result).squeeze()
            adversarial_loss = D_criterion(D_result, y_real)
            # perception loss
            perception_loss = G_criterion(VGG_feature_model(G_result), VGG_feature_model(hr_images))
            G_train_loss = image_loss + 1e-3 * adversarial_loss + 2e-6 * perception_loss
            # step
            G_train_loss.backward()
            G_optimizer.step()


            G_train_loss_sum += G_train_loss.cpu().item()
            D_train_loss_sum += D_train_loss.cpu().item()
            n += temp_batch_size
            whole_batch_count += 1
            batch_count += 1
            G_temp_loss = G_train_loss_sum / whole_batch_count
            D_temp_loss = D_train_loss_sum / whole_batch_count
            Loss_list.append(G_train_loss.item())
            logging.info('-epoch %d, batch_count %d, img nums %d, G_loss temp %.4f, D_loss temp %.4f, time %.1f sec,'
                  % (epoch + 1, whole_batch_count, n, G_temp_loss, D_temp_loss, time.time() - start))
            print('-epoch %d, batch_count %d, img nums %d, G_loss temp %.4f, D_loss temp %.4f, time %.1f sec'
                  % (epoch + 1, whole_batch_count, n, G_temp_loss, D_temp_loss, time.time() - start))

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
                      % (epoch + 1, n, temp_psnr_test, G_temp_loss, time.time() - start))
                print('---epoch %d, img nums %d, psnr_mean %.4f, loss %.4f, time %.1f sec---'
                      % (epoch + 1, n, temp_psnr_test, G_temp_loss, time.time() - start))

        psnr_test_list.append(temp_psnr_test)

        result_path = model_save_path / ("epoch_" + str(epoch) + "_lr_" + str(lr_G) +"_test_result.csv")
        create_csv(result_path, test_result_list)

        torch.save(net.state_dict(),
                   model_save_path / ("_epoch_" + str(epoch) + "_lr_" + str(lr_G) + "_" + str(
                       time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".pth"))

def run():
    # main function described in report
    train_dataset = amls_dataset(train_datapath, "training")
    test_dataset = amls_dataset(test_datapath, "test")

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)

    # perception model
    VGG_model = vgg19(pretrained=True)
    VGG_feature_model = nn.Sequential(*list(VGG_model.features)[:-1]).eval()
    for param in VGG_feature_model.parameters():
        param.requires_grad = False

    # # generator
    # if residual == True:
    #     net = FSRCNN_Residual(num_channels=3)
    # else:
    #     net = FSRCNN(num_channels=3)

    net = Generator(2)

    # discriminator
    teacher = Discriminator()

    if pretrain == True and residual == False:
        net.load_state_dict(torch.load(pretrained_net_path))
        print("---load pretrained generator models")
        teacher.load_state_dict(torch.load(pretrained_teacher_path))
        print("---load pretrained discrimintor models")

    # G_optimizer = optim.Adam([
    #     {'params': net.first_part.parameters(), 'lr': lr_G},
    #     {'params': net.mid_part.parameters(), 'lr': lr_G},
    #     {'params': net.last_part.parameters(), 'lr': lr_G * 0.1}
    # ], lr=lr_G, betas=(momentum, 0.999), weight_decay=weight_decay)

    G_optimizer = optim.Adam(net.parameters(), lr=lr_G, betas=(momentum, 0.999), weight_decay=weight_decay)

    D_optimizer = optim.Adam(teacher.parameters(), lr=lr_D, betas=(momentum, 0.999), weight_decay=weight_decay)

    G_loss = torch.nn.MSELoss()
    D_loss = torch.nn.BCEWithLogitsLoss()

    train(net, teacher, VGG_feature_model, train_iter, test_iter, G_loss, D_loss, G_optimizer, D_optimizer, num_epochs=epoch_num)

    plot_save(Loss_list, psnr_test_list)

    # torch.save(net.state_dict(),
    #            model_save_path / (task + "_epoch_" + str(epoch) + "_lr_" + str(lr) + "_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) +".pth"))

if __name__ == '__main__':
    run()
