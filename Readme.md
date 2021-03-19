# Cartoonification Task
## Description 
It's required to develope a model for an entertainment app in which users will provide pictures for you to apply a cartoon effect on (Make them in the style of cartoons).
Your ideal input would be a frontal image of the user and you should provide the outputted image with the style applied.

for such a problem we will be using a specific type of deep learning Networks: GANs (Generative Adversarial Networks).

# Introducion to GANs
Generative adversarial networks (GANs) are an exciting recent innovation and an unsupervised learning technique in machine learning. GANs are generative models: they create new data instances that resemble your training data. For example, GANs can create images that look like photographs of human faces, even though the faces don't belong to any real person.

GANs achieve this level of realism by pairing a generator, which learns to produce the target output, with a discriminator, which learns to distinguish true data from the output of the generator. The generator tries to fool the discriminator, and the discriminator tries to keep from being fooled.
for example if we want to build an animal idenrifier, A generative model could generate new photos of animals that look like real animals, while a discriminative model could tell a dog from a cat. GANs are just one kind of generative model.

More formally, given a set of data instances X and a set of labels Y:

- Generative models capture the joint probability p(X, Y), or just p(X) if there are no labels.
- Discriminative models capture the conditional probability p(Y | X).

Note that this is a very general definition. There are many kinds of generative model. GANs are just one kind of generative model.

![alt text](https://developers.google.com/machine-learning/gan/images/gan_faces.png)

**Figure 1:** Images generated by a [GAN created by NVIDIA.](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf)

#### why GANs?

there are several reasons why GANs are so important, they are very useful to tackle the following problems:

-  Generation of synthetic training data for machine learning models in case training data is insufficient or collecting it is too costly.
- Generation of human faces, objects in 2D and 3D, realistic photographs, anime characters, and music.
- Media Translations like image-to-image translations, semantic image-to-photo translations, and text-to-image translations.
-  Editing photographs by denoising images and enhancing the existing image data using super-resolution, photo blending.
-  Creating Image style transfers or audio style transfers.

#### GANs Structure.

figure 2 here represents the structure of the whole system, Both the generator and the discriminator are neural networks. The generator output is connected directly to the discriminator input. Through backpropagation, the discriminator's classification provides a signal that the generator uses to update its weights.

![](https://developers.google.com/machine-learning/gan/images/gan_diagram.svg)

**Figure 2:** General GANs structure.

#### GAN Variations

Researchers continue to find improved GAN techniques and new uses for GANs. Here's a sampling of GAN variations to give you a sense of the possibilities.

here are some of the different types of GANs networks:
- **Progressive GANs:** This technique allows the GAN to train more quickly than comparable non-progressive GANs, and produces higher resolution images.
- **Conditional GANs:** Conditional GANs train on a labeled data set and let you specify the label for each generated instance. instead of modeling the joint probability P(X, Y), conditional GANs model the conditional probability P(X | Y)
- **Image-to-Image Translation:**Image-to-Image translation GANs take an image as input and map it to a generated output image with different properties. In these cases, the loss is a weighted combination of the usual discriminator-based loss and a pixel-wise loss that penalizes the generator for departing from the source image.
- **CycleGAN:** CycleGANs learn to transform images from one set into images that could plausibly belong to another set. For example, a CycleGAN produced the righthand image below when given the lefthand image as input. It took an image of a horse and turned it into an image of a zebra.
![](https://developers.google.com/machine-learning/gan/images/cyclegan.png)

The training data for the CycleGAN is simply two sets of images (in this case, a set of horse images and a set of zebra images). The system requires no labels or pairwise correspondences between images, **which is actually the case of our problem** here since there's no enough paired data for transforming a real face to a cartoonized one, and generating These datasets can be difficult and expensive to prepare, which is the reason why using cycleGANs here would be better as it's easier to provide unpaired datasets.

## Applied Solution

there are several state of the art CycleGANs models in the industry that differ in performance in terms of speed and output accuracy, in this repo we will be using 2 different models to tackle our problem, which are: AnimeGANv2 and StyleGAN.

**for AnimeGANv2:** the model was trained from scratch to generate the weights, and then they were used for testing and generating the output.

**for StyleGAN:** the pretrained model was used to generate the results, the training process can be done similarly to AnimeGAN.

this is a sample of giving both models a set of test images and the generated results from each model.

note that both model have two very different style to apply on the images (AnimeGANv2 turns the images to Anime-like images aqcuired from Anime films while cycleGAN turns them to toons like in disney and pixar) which is not due to the different model structure but due to the different datasets distribution used for training, which actually shows how GANs work, that the model tries to learn features from a specific set of images to turn other images to that set.
for more information about each model, please navigate to its directory and check the readme.



| Original Images | AnimeGANv2 | StyleGAN   |
|     ---    |     ---    |            ---     |
| <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/dataset/test/Omar/1.jpg" alt="" width="900" height="300"/> | <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/results/Omar_Results/1.jpg" alt="" width="900" height="300"/> | ![](https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/StyleGAN2/generated/1_01-toon.jpg) |
| <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/dataset/test/Omar/2.jpg" alt="" width="850" height="300"/> | <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/results/Omar_Results/2.jpg" alt="" width="850" height="300"/> | ![](https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/StyleGAN2/generated/2_01-toon.jpg) |
| <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/dataset/test/Omar/3.jpg" alt="" width="800" height="300"/> | <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/results/Omar_Results/3.jpg" alt="" width="800" height="300"/> | ![](https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/StyleGAN2/generated/3_01-toon.jpg) |
| <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/dataset/test/Omar/4.jpg" alt="" width="800" height="250"/> | <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/results/Omar_Results/4.jpg" alt="" width="800" height="250"/> | ![](https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/StyleGAN2/generated/4_01-toon.jpg) |
| <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/dataset/test/Omar/5.jpg" alt="" width="800" height="300"/> | <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/results/Omar_Results/5.jpg" alt="" width="800" height="300"/> | ![](https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/StyleGAN2/generated/5_01-toon.jpg) |
| <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/dataset/test/Omar/6.jpg" alt="" width="870" height="300"/> | <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/results/Omar_Results/6.jpg" alt="" width="870" height="300"/> | ![](https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/StyleGAN2/generated/6_01-toon.jpg) |
| <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/dataset/test/Omar/7.jpg" alt="" width="700" height="250"/> | <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/results/Omar_Results/7.jpg" alt="" width="700" height="250"/> | ![](https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/StyleGAN2/generated/7_01-toon.jpg) |
| <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/dataset/test/Omar/Omar.jpg" alt="" width="400" height="300"/> | <img src="https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/AnimeGANv2/results/Omar_Results/Omar.jpg" alt="" width="400" height="300"/> | ![](https://github.com/OmarM-Abdallah/Cartoonification-Task/blob/main/StyleGAN2/generated/Omar_01-toon.jpg) |


### Original Repos
**AnimeGANv2:**[here](https://github.com/TachibanaYoshino/AnimeGANv2)

**StyleGAN:**[here](https://github.com/justinpinkney/stylegan2)



