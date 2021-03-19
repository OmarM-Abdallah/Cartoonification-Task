
# AnimeGANv2
AnimeGANv2 is a GAN model that is based off the original [CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf) and implemented using tensorflow. CartoonGAN is built to apply Photo Cartoonization, and AnimeGAN achieves the same target but in a different distribution of cartoon which is anime.

### Contents
- Introduction
- How It Works
- Installation Requirements
- Usage Guide

### Introduction
The approach proposed combines neural style transfer and generative adversarial networks (GANs) to achieve the task at hand. one of the most advantages of AnimeGAN is that the parameters of the network require less memory capacity than other architectures require, which makes it faster and easier to use.

the paper introduces the pre-trained VGG19 as the perceptual network
to obtain the L1 loss of the deep perceptual features of the generated images and original photos.

Before AnimeGAN starts training, an initialization training on the generator is done to make the training of AnimeGAN easier and more stable. A large number of experimental results show that AnimeGAN can quickly generate higher-quality anime style images.

### How It Works

Figure 1 shows the model architecture of AnimeGAN, which consists of a generator and a discriminator network each having different type of blocks.


![](https://www.programmersought.com/images/99/06b5aed9294b5ea7597b2d4c730783f3.png)

**Figure 1**: The architecture of the generator and the discriminator in the proposed AnimeGAN.

in addition to the model training and prediction process, several steps for pre and post-processing are imlplemented to improve the output results.

**pre-training:**
- a step of edge smoothing is done to smooth the edges in each image, which makes the feature extraction easier, this is done using gaussian filter and then the image is resized for further processing.

- another pre processing step is to apply data mean to the images dataset which is used to calculate the three-channel(BGR) color difference of the entire style data, and these difference values are used to balance the effect of the tone of the style data on the generated image during the training process.

**post-training:**
- Extracting the weights of the generator and discarding the discriminator weights as they are no longer needed, we just need the generator weights to use for testing and generating results.

for more info about AnimeGANv2 and its architecture refer to the paper [here.](https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/AnimeGANANovelLightweightGAN.pdf)

### Installation Requirements

- python 3.6
- tensorflow-gpu
  - tensorflow-gpu 1.8.0 (ubuntu, GPU 1080Ti or Titan xp, cuda 9.0, cudnn 7.1.3)
  - tensorflow-gpu 1.15.0 (ubuntu, GPU 2080Ti, cuda 10.0.130, cudnn 7.6.0)
- opencv
- tqdm
- numpy
- glob
- argparse

### Usage Guide

##### 1. Download vgg19  

download the pre-trained VGG19 which is used as the perceptual network as mentioned before and move it to vgg19_weight directory.
  > [vgg19.npy](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/vgg16%2F19.npy) 

##### 2. obtain Train/Val Photo dataset 

use your own personal dataset or download the original dataset used by authors and move it to the dataset direcctory to be used for training, there are mainly four folders here: 
- the train_photo folder should contain the real world images that are to be animated.
- the second one is a style folder (several ones exist here with different styles) that the model uses the images in it to apply the style of the provided datasets to the real world images in the train_photo folder.
- val photos are used for validation.
- test photos are the ones to test the model with after training is done.
  > [Download dataset used by authors](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1) 



##### 3. Do edge_smooth  
  > `python3 edge_smooth.py --dataset Omar --img_size 256`  
  
##### 4. Calculate the three-channel(BGR) color difference  
  >  `python3 data_mean.py --dataset Omar`  
  
##### 5. Train  
  >  `python3 main.py --phase train --dataset Omar --data_mean [13.1360,-8.6698,-4.4661] --epoch 101 --init_epoch 10`  
  >  For light version: `python3 main.py --phase train --dataset Omar --data_mean [13.1360,-8.6698,-4.4661]  --light --epoch 101 --init_epoch 10`  
  
##### 6. Extract the weights of the generator  
  >  `python3 get_generator_ckpt.py --checkpoint_dir  ../checkpoint/AnimeGAN_Omar_lsgan_300_300_1_2_10_1  --style_name Omar`  

##### 7. Inference      
  > `python3 test.py --checkpoint_dir  checkpoint/generator_Omar_weight  --test_dir dataset/test/Omar --style_name Omar/Omar_Results` 

##### Important Notes: 
- **for steps 3-7:** adjust the commands arguments in the following steps to suit your usage.
- the model was trained using google colab , which doesn't provide a consistant GPU, so unfortunately it wasn't possible to obtain relevant statistics of the training process in terms of training time or for testing in terms of FPS, however you can refer to the [paper](https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/AnimeGANANovelLightweightGAN.pdf) as it does provide these information.
