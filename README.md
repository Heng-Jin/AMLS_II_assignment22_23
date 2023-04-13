# AMLS_II_assignment22_23
AMLS_II_assignment22_23
/ UCL ELEC0135 course final assignment by Heng Jin 22212102

This readme is to illustrate the structure of the codes and precedures to train-test the model.

### This is variant subtask of the NTIRE 2017 challenge.
A super resolution challenge, the model training can only use limited 800 pair of images.

Variant:

x8 images (maximum size 255x255) is defined as LR image

x4 images (maximum size 510x510) is defined as HR image

### FSRCNN and SRGAN is implemented and fused in this tasks.  

To achieve fast development of codes, I used Nvidia GPU to accelerate the runtime. 

Under the environment of NVIDIA Tesla V100,
Nvidia Driver Version: 510.47.03, CUDA Version 11.6, 
it takes roughly 10 secs per epoch (800 training images) for Training FSRCNN in the batchsize of 64, 100 sec per epoch for SRGAN in the batchsize of 8.

### Python libraries used
- pytorch
- numpy
- matplotlib
- pillow
- csv
- pathlib
- logging

All these are frequently used libraries, you can install them manually.
Or run the command : pip install -r requirements.txt to on python 3.6 
virtual environment to configure relevant libraries.

For pytorch, please check official website to install it correctly.
<https://pytorch.org/get-started/locally/>

### run the code

First please download the dataset in the link <https://data.vision.ee.ethz.ch/cvl/DIV2K/>, training dataset and validation dataset of X8 and X4 are required.

Please unzip and put them under the /data/ folder, like the program structure shows.

To run the training code of FSRCNN: python train_fsrcnn.py

To generate the HR images, please run : python img_solver.py --input lr_image_path --model model_path

### Program structure
-- AMLS_II_assignment22_23 

&emsp; -- data

&emsp;&emsp;  -DIV2K_train_LR_bicubic

&emsp;&emsp;&emsp;  -X8

&emsp;&emsp;&emsp;  -X4

&emsp;&emsp;  -DIV2K_valid_LR_bicubic

&emsp;&emsp;&emsp;  -X8

&emsp;&emsp;&emsp;  -X4

&emsp; train_fsrcnn.py

&emsp; train_srgam.py

&emsp; train_generator.py

&emsp; dataset.py

&emsp; model.py

&emsp; . . . .



### Program run instruction
The train.py defines the training and validation pipeline of the model. 
The inference of the validation dataset will be implemented after each 
training epoch. The model will be saved after each epoch as well. 
All the outputs of each training will be saved into a separate folder.

&emsp; train_fsrcnn.py : train FSRCNN or Residual-FSRCNN

&emsp; train_srgam.py : train SRGAN-FSRCNN 

&emsp; train_generator.py : train RGB 3 channel FSRCNN (generator of SRGAN-FSRCNN) alone

&emsp; train_discriminator.py : train discriminator of SRGAN-FSRCNN alone

The model.py defines the model structure of each model, involving basic 
FSRCNN, FSRCNN with skip connection, and discriminator of SRGAN.

&emsp; model.py : model of FSRCNN, Residual-FSRCNN, RGB 3 channel FSRCNN

&emsp; srgan.py : model of discriminator of SRGAN-FSRCNN, SRResNet

The dataset.py defines the Class Dataset to serve LR im-ages and HR images for each 
iteration of model training and inference. For basic FSRCNN, images are converted to 
YCbCr color format and only the tensor in the Y channel will be returned, according to 
the design of FSRCNN. For RGB 3-channel FSRCNN, tensors with all 3 channels will be returned.

&emsp; dataset.py : Dataset for Y channel input models

&emsp; dataset_gan.py : Y channel for RGB 3 channel input models

### Image result

[//]: # (<img src="D:\UCL_codes\0135\git_submission\AMLS_II_assignment22_23\image results\Origin.png" height="135"/>)

[//]: # (<img src="D:\UCL_codes\0135\git_submission\AMLS_II_assignment22_23\image results\FSRCNN.png" height="135">)

[//]: # (<img src="D:\UCL_codes\0135\git_submission\AMLS_II_assignment22_23\image results\Pretrained_FSRCNN.png" height="135">)

[//]: # (<img src="D:\UCL_codes\0135\git_submission\AMLS_II_assignment22_23\image results\Residual-FSRCNN.png" height="135"/>)

[//]: # (<img src="D:\UCL_codes\0135\git_submission\AMLS_II_assignment22_23\image results\RGB 3channel FSRCNN.png" height="135"/>)

[//]: # (<img src="D:\UCL_codes\0135\git_submission\AMLS_II_assignment22_23\image results\SRGAN-FSRCNN.png" height="135"/>)


<img src="https://github.com/Heng-Jin/AMLS_II_assignment22_23/blob/main/image%20results/Origin.png" height="135"/><img src="https://github.com/Heng-Jin/AMLS_II_assignment22_23/blob/main/image%20results/FSRCNN.png" height="135"/><img src="https://github.com/Heng-Jin/AMLS_II_assignment22_23/blob/main/image%20results/Pretrained_FSRCNN.png" height="135"/><img src="https://github.com/Heng-Jin/AMLS_II_assignment22_23/blob/main/image%20results/Residual-FSRCNN.png" height="135"/><img src="https://github.com/Heng-Jin/AMLS_II_assignment22_23/blob/main/image%20results/RGB%203channel%20FSRCNN.png" height="135"/><img src="https://github.com/Heng-Jin/AMLS_II_assignment22_23/blob/main/image%20results/SRGAN-FSRCNN.png" height="135"/>

&emsp; Origin Image &emsp; FSRCNN result &emsp; Pretrained FSRCNN &emsp; Residual-FSRCNN &emsp; RGB 3channel FSRCNN &emsp; SRGAN-FSRCNN 



