
# Diffusion model from scratch

## 1. Background

Diffusion models have become the state-of-the-art approach for generating images, videos, and music. They power popular tools such as DALL-E (OpenAI), Stable Diffusion (Stability AI), FireFly (Adobe), Diffusers Library (Hugging Face), and Midjourney (Midjourney, Inc). They have also found applications in many other domains, including 3D modelling, medical imaging, drug discovery, molecular design, and more.

Diffusion models first appeared in a 2015 paper titled "Deep unsupervised learning using Nonequilibrium Thermodynamics" by Jascha Sohl-Dickstein et al. While the paper was theoretically significant, results were limited to toy datasets. It received little attention at the time, as research was largely focused on GANs.

In 2020, Jonathan Ho et al. published a paper titled "Denoising Diffusion Probabilistic Models" (DDPMs). The main contribution of this paper was to transform the ideas from the 2015 work into a state-of-the-art technique for image generation. The paper quickly gained traction and is now one of the most cited works in the field, with 25,000+ citations as of 2025.

Because it required many iterative steps, the sampling method used in the DDPM paper was very slow (taking up to several hours for a single image). Shortly after the DDPM paper came out, Jiaming Song et al. published "Denoising Diffusion Implicit Models" (DDIMs), which introduced a much faster sampling technique. This new approach made diffusion models a practical solution for image generation.

Subsequent papers introduced various improvements that ultimately made diffusion models a superior solution to GANs, producing higher-quality images while being significantly easier to train.

## 2. Project goal

The goal of this project was to recreate the DDPM model from the 2020 paper by Johnathan Ho et al., train it, and generate samples using both the DDPM and DDIM methods.

Johnathan Ho posted on Github the TensorFlow 1 code they used for their research, at this URL provided in the paper:

[JohnathanHo/DDPM](https://github.com/hojonathanho/diffusion)

Many PyTorch implementations of DDPM are now available on GitHub, including:

- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch?utm_source=chatgpt.com)

- [openai/improved-diffusion](https://github.com/openai/improved-diffusion?utm_source=chatgpt.com)

- [mattroz/diffusion-ddpm](https://github.com/mattroz/diffusion-ddpm?utm_source=chatgpt.com)

- [cloneofsimo/minDiffusion](https://github.com/cloneofsimo/minDiffusion?utm_source=chatgpt.com)

Because my goal was to gain a deep understanding of DDPM models, I only relied on the original 2020 DDPM and DDIM papers, and some of the works they reference. I only looked at Ho's code on GitHub when important implementation details were not specified in the paper.

I initially planned to use the CIFAR-10 dataset, as in the DDPM paper. However, during the course of the project, I realized that I did not have sufficient GPU resources at my disposal to train the same U-Net. Therefore, I used the MNIST dataset with a scaled-down U-Net.


## 3. Source code

All the code is in TensorFlow 2. Custom Keras layers and models are used for the U-Net and diffusion models.

The code is in the *./src* directory and is organized as shown below.

```
    src
     |     
     ├── u_net.py                   # U-Net model
     |
     ├── u_net_debug.py             # Same as u_net.py with prints and assertions for architecture debugging
     |
     ├── train.py                   # Diffusion model training script
     |
     ├── sample.py                  # Sampling training script (DDPM and DDIM)
     |
     ├── mnist                      # Directory containing examples of generated MNIST images
     |
     └── cifar10                    # Directory containing examples of generated CIFAR-10 images
```

## 4. Information missing in the DDPM paper

The DDPM paper describes in great detail the theory underlying DDPM models. It also provides training and sampling algorithms, which are straightforward to implement.

However, the information provided about the U-Net used to predict noise in the images is largely limited to the following paragraph:

```
B. Experimental details

Our neural network architecture follows the backbone of PixelCNN++ [52], which is a U-Net [48] based on a Wide ResNet [72]. We replaced weight normalization [49] with group normalization [66] to make the implementation simpler. Our 32 x 32 models use four feature map resolutions (32 x 32 to 4 x 4), and our 256 x 256 models use six. All models have two convolutional residual blocks per resolution level and self-attention blocks at the 16 x 16 resolution between the convolutional blocks [6]. Diffusion time t is specified by adding the Transformer sinusoidal position embedding [60] into each residual block.

```

This description is not sufficient to recreate the U-Net model that was used. Therefore, I had to reverse-engineer Ho's code on Github to obtain the missing implementation details.

## 5. U-Net model architecture

In Ho's code on Github, the 32 x 32 U-Net is implemented as shown in Figure 1. This is the U-Net they used for the CIFAR-10 dataset. The model has parameters to configure it for the 256 x 256 images of the Celeb-A and LSun datasets.

![](pictures/unet_cifar10.PNG)


The model follows the "classic" U-Net architecture:

- U-shaped architecture with a contracting path (encoder) and an expanding path (decoder)

- Skip connections that concatenate features from the contracting path to the corresponding layers in the expanding path, allowing the network to combine low-level and high-level features.

Figure 2 shows the structure of the ResNet block as it is implemented in Ho's code.

![](pictures/resnet_block.PNG)

It was clearly inspired by Wide ResNet and PixelCNN++, as mentioned in the paper.

It has the following key features:

- The residual connection adds the block’s input to its output after the block’s transformations. If the block input and output shapes are the same, it is a straight connection. If they are different, a 1 x 1 convolution layer (Network-in-Network) layer is inserted in the connection to make the shapes compatible for addition.

- Group normalization is done *before* convolution (pre-normalization).

- The timestep position embedding is passed through an MLP and added to the output of the first convolution. This allows the model to be aware of the current step in the diffusion process.

- Activations are SiLU (Swish) activations.

- A dropout layer is inserted between the activation and convolution layers of the second sub-block.


## 4. MNIST U-Net

I did not have access to sufficient GPU resources to reproduce the CIFAR-10 results. Therefore, I used MNIST instead with the scaled-down U-Net model that is shown in Figure 3.

![](pictures/unet_mnist.PNG)

I made the following changes to the U-Net in Ho's code:

- Input images resized from 28 x 28 to 32 x 32, needed to have 4 resolution sizes
- Number of base channels (channels that the input convolution outputs) reduced from 128 to 64.
- Only 1 ResNet block per up/down stage instead of 2
- Attention at the 8 x 8 resolution instead of 16 x 16
- Output images resized from 32 x 32 to 28 x 28

## 4. Training setup

The training setup the DDPM authors used is described in section *B. experimental details*

I used the same setup as they did for the CIFAR-10 dataset:

- Data augmentation: random horizontal flips

- Timesteps: 1000

- Dropout rate: 0.1

- Optimizer: Adam with learning rate 2e-4

- Batch size: 128

- EMA decay factor: 0.9999

DDPM used a linear beta schedule. I used a cosine schedule instead, which was introduced later and proved a superior solution.


## 5. MNIST sampling

Examples of samples obtained with the DDPM sampling method are shown in figure 4. The images are shown at different time steps of the sampling process. The quality of the generated images is quite good, and so is diversity in a batch of images.

![](pictures/ddpm_samples.PNG)

Examples of samples obtained with the DDIM sampling method are shown in Figure 5. Only 50 steps were used to obtain these images, using a completely deterministic method. Image quality and diversity are quite good.

![](pictures/ddim_samples.PNG)

On Google Colab using a T4 GPU, generating a batch of 128 images takes 4:45min wall clock with the DDPM sampling method. It only takes 16sec with the DDIM method, which is 18x faster.

