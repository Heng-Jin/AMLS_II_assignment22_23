import os
import csv
from matplotlib import pyplot as plt

'''
This python script calculate the max PSNR of the epoch in each folder.
The Curves of PNSR and Loss will be generated and saved in /figure_save_place/
'''

folder_path = os.getcwd()

img_path = os.path.join(os.getcwd(), 'figure_save_place')
if not os.path.exists(img_path):
    os.makedirs(img_path)

def plot_save(loss_list, acc_list, plot_save_path):
    '''
    plot temporary loss of training and accuracy of test dataset after each epoch training
    Args:
        loss_list: list of loss value of each iteration
        acc_list: list of PSNR value of each epoch

    Returns: nothing

    '''
    # plot temporary loss of training and accuracy of test dataset after each epoch training
    x1 = range(len(acc_list))
    x2 = range(len(loss_list))
    y1 = acc_list
    y2 = loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '.-')
    plt.title('Validation PSNR vs. epoches')
    plt.ylabel('Validation PSNR')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Training loss vs. iteration')
    plt.ylabel('Training loss')
    # plt.show()
    plt.savefig(plot_save_path,dpi=600, bbox_inches='tight')
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure
    plt.close()


for dirpath, dirnames, filenames in os.walk(folder_path):
    acc_list = []
    loss_list = []
    # print(dirpath, dirnames, filenames)
    for filename in filenames:
        if filename == 'acc_list.csv':
            csv_path = os.path.join(dirpath, filename)

            best_acc = 0.0
            best_model = ''
            best_epoch = 0
            with open(csv_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for i, row in enumerate(reader):
                    value = float(row[0]) * 2.0
                    acc_list.append(value)
                    if value > best_acc:
                        best_acc = value
                        best_epoch = i
                        best_model = dirpath

            print("acc_list dir is: ", best_model)
            print("max acc epoch is: ", best_epoch)
            print("the max acc is: ", best_acc)
            print("")

        if filename == 'loss_list.csv':
            csv_path = os.path.join(dirpath, filename)

            with open(csv_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for i, row in enumerate(reader):
                    value = float(row[0])
                    loss_list.append(value)

    if len(loss_list) != 0:
        # print(os.path.basename(dirpath))
        plot_save_path = os.path.join(img_path, (str(os.path.basename(dirpath))+'_acc_loss.png'))
        plot_save(loss_list, acc_list, plot_save_path)
