
# Diffusion model from scratch using research papers

"What I cannot create, I do not understand."
— **Richard Feynman**

## 1. Background

Diffusion models have become the state-of-the-art approach for generating images, videos, and audio. They power popular tools such as DALL-E 3, Stable Diffusion, Adobe Firefly, and Midjourney. They also found applications in many other domains, including 3D modeling, medical imaging, drug discovery, and molecular design.

While the foundational concept of diffusion-based generation was introduced in 2015 by Sohl-Dickstein et al., the modern framework was established in 2020 by Jonathan Ho et al. in their seminal paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPMs), which is now the most cited work in the field with 20,000+ citations as of 2025.

Shortly after, Jiaming Song et al. introduced a new sampling technique that substantially reduced generation times by allowing deterministic, non-Markovian denoising: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (DDIMs).

In 2022, Robin Rombach et al. introduced diffusion models that operate in a compressed latent space rather that in the pixel space, enabling efficient scaling: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (LDMs).

Ultimately, diffusion models surpassed GANs by producing higher-quality, more diverse samples while offering significantly greater training stability.


## 2. Project goals

The goal of this project was to recreate the DDPM model from the 2020 paper by Jonathan Ho et al., train it, and generate images using both the DDPM (probabilistic) and DDIM (deterministic) sampling methods.

While excellent PyTorch implementations of DDPM already exist (such as those by [lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch), [OpenAI](https://github.com/openai/improved-diffusion), [mattroz](https://github.com/mattroz/diffusion-ddpm), and [cloneofsimo](https://github.com/cloneofsimo/minDiffusion)), high-quality, modern TensorFlow/Keras implementations are far less common. For that reason, I chose to implement my model in TensorFlow.

The math underlying diffusion models, the training and sampling algorithms, and the training setup are well-documented in the DDPM paper. However, the architecture of the U-Net is only described at a high-level (section "B. Experimental details"). Because my goal was to directly compare my FID (Fréchet Inception Distance) scores with those reported in the paper, an exact architecture match was required. As a last resort, I looked for the missing details in [Jonathan Ho's GitHub repository](https://github.com/hojonathanho/diffusion).

I first used the MNIST dataset with a light-weight U-Net as a "pipe-cleaner". The shorter runtimes facilitated the validation of training and sampling flows, and FID calculations.

Then, I trained a model on CIFAR-10 using the same U-Net as in the DDPM paper. I did not attempt to train models on the LSUN or CelebA-HD datasets due to GPU hardware constraints.


## 3. Source code and Python packages

The code for this project is in the **./src** directory and is organized as shown below.

```
   src
    |     
    ├── u_net.py                   # U-Net model
    |
    ├── train_mnist.py             # Train diffusion model on MNIST dataset
    |
    ├── train_cifar10.py           # Train diffusion model on CIFAR-10 dataset
    |
    ├── compute_fid.py             # Compute FID score
    |
    ├── view_gen_images.py         # Generate images and display them
    |
    └── utils.py                   # Utilities and shared functions
```

See file **requirements.txt** for the list of Python packages I used.

## 4. U-Net model architecture

In Ho's code on Github, the U-Net used for the 32 x 32 images of the CIFAR-10 dataset is implemented as shown in Figure 1. The model has parameters to configure it for the 256 x 256 images of the CelebA and LSUN datasets.

![](pictures/unet_cifar10.png)

The model follows the "classic" U-Net architecture:

- U-shaped architecture with a contracting path (encoder) and an expanding path (decoder)

- Skip connections that concatenate features from the contracting path to the corresponding layers in the expanding path, allowing the network to combine low-level and high-level features

Figure 2 shows the structure of the ResNet block as it is implemented in Ho's code.

![](pictures/resnet_block.png)

The residual block was clearly inspired from Wide ResNet and PixelCNN++, as mentioned in the paper, and has the following key features:

- The residual connection adds the block’s input to its output after the block’s transformations. If the block input and output have the same number of channels, it is a straight connection. If they are different, a Network-in-Network (1 x 1 convolution) layer is inserted in the connection to make the shapes compatible for addition.

- Group normalization is done *before* convolution (pre-normalization).

- The timestep position embedding is passed through an MLP and added to the output of the first convolution. This allows the model to be aware of the current step in the diffusion process.

- Activations are SiLU (Swish) activations.

- A dropout layer is inserted between the normalization and convolution layers of the second sub-block.

In appendix B "Experimental Details" of the DDPM paper :

```
Our CIFAR-10 model has 35.7 million parameters, and our LSUN and CelebA-HQ models have 114 million
parameters. We also trained a larger variant of the LSUN Bedroom model with approximately 
256 million parameters by increasing filter count.
```

In the CIFAR-10 configuration (Figure 1), my U-Net model has 35.9M parameters. I was not able to explain the discrepancy with the 35.7M number given in the paper.

## 5. MNIST diffusion model

### 5.1 U-Net

The U-Net I used for the MNIST dataset is shown in Figure 3.

![](pictures/unet_mnist.png)

I made the following changes to the CIFAR-10 U-Net used in the DDPM paper:

- Input images padded from 28 x 28 to 32 x 32 to enable 4 resolution sizes
- Number of base channels (channels that the input convolution outputs) reduced from 128 to 64
- Only 1 ResNet block per up/down stage instead of 2
- Attention at the 8 x 8 resolution instead of 16 x 16
- Output images cropped from 32 x 32 to the original 28 x 28 size

 With only 4.9M parameters, the MNIST U-Net is much smaller than the CIFAR-10 U-Net that has 35.9M parameters.

### 5.2 Training setup

I used the training setup described in Appendix B "Experimental details" of the DDPM paper:

- Timesteps: 1000
- EMA decay factor: 0.9999
- Dropout rate: 0.1
- Optimizer: Adam with learning rate 2e-4
- Batch size: 128

I trained the model for 250 epochs.

### 5.3 FID score

Using 10,000 images from the MNIST training set and DDIM sampling with 50 steps, the FID score is 29.8. 

This is not a good score, but it was expected given the small size of the U-Net. The goal was not to obtain state-of-the-art results.

### 5.4 Visualizing generated images

Examples of images generated with the DDPM sampling method are shown in Figure 4. The images are shown at different timesteps of the reverse process.

![](pictures/mnist_ddpm_samples.png)

Examples of images obtained using DDIM sampling with 50 steps are shown in Figure 5.

![](pictures/mnist_ddim_samples.png)

Low-quality images are shown on the last row, where it is impossible to identify the digits with reasonable confidence.


## 6. CIFAR-10 diffusion model

### 6.1 U-Net

For CIFAR-10, I used the exact same U-Net as in Ho's code (Figure 1).

### 6.2 Training

I used the training setup described in appendix B "Experimental results" of the DDPM paper. It is the same setup as for MNIST, with the addition of random horizontal flips to increase image diversity.

Ho et al. specified that they trained their model for ~800k steps with a batch size of 128, which represents 2048 epochs, so I did the same.

The loss values for each epoch are shown in Figure x. No more improvement is visible when the training ends at epoch 2048.

Runtime on an A100 GPU was 48sec/epoch on an A100 GPU, so the training took ~27 hours to run.

![](pictures/mse_loss.png)

### 6.3 FID scores

Like Ho et al., I used the 50,000 images of the CIFAR-10 training set as the reference distribution.

I computed the FID at epoch 1200 of the training, which is about half-way to Ho et al.'s 800K steps. Using deterministic DDIM sampling with 200 steps, the computed value is 4.94.

FID values at the end of the training at epoch 2048 are shown in the table below for different numbers of DDIM steps, together with the runtimes on an A100 GPU generating batches of 2000 images.


|  Denoising steps  |  Sampling method   |   FID  |  Runtime (hh:mm)  |
|-------------------|-----------|-----------|---------------------|
|      50           |    DDIM   |   5.77    |         0:20        |
|     100           |    DDIM   |   4.53    |         0:41        |
|     200           |    DDIM   |   4.11    |         1:22        |
|    1000           |    DDPM   |           |                     |

As expected:

- The FID improves with the number of steps.
- Because they are dominated by the number of forward passes through the U-Net, runtimes increase linearly with the number of steps.

Note that the 200-step-DDIM FID decreased from 4.94 after 1200 training epochs to 4.11 after 2048 epochs.

Ho et al. reported an FID value of 3.17 using the 50,000 images of the CIFAR-10 training set as the reference distribution (Table 1 in the paper).

### 6.4 Visualizing generated images

Figure 7 shows examples of images obtained using the DDPM sampling method.

![](pictures/cifar10_ddpm_samples.png)

Figure 8 shows images generated using the DDIM sampling method with 200 steps.

Examples of generative "hallucinations" are shown on the last row of images:

- A cat with a red mouse on top of its head
- An entirely red dog lying on the ground
- A pink cat
- A deer with front legs shaped like horns

The images were obtained using only 25 DDIM steps, which corresponds to an FID of x.

![](pictures/cifar10_ddim_samples.png)


## 7. Conclusion

Recreating the diffusion model from the landmark DDPM paper by Jonathan Ho et al. bridged the gap between the theoretical generative equations and the intricacies of the U-Net architecture, training and evaluation procedures, and interpretation of results.

It was absolutely fascinating to observe images progressively emerging from noise during the reverse process!
