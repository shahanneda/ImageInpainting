# ImageInpainting
### An implemention of the classic paper "ContextEncoders" along with some of my own improvments

### Abstract:

In this project, we attempt to create an image inpainting workflow that takes in an image with a cutout through the center, and tries to infill this area in the most realistic way possible by generating an infilling and integrating it to the image. We do so by combining an autoencoder based approach to generate the infilling, along with a more traditional CV approach to integrate the infilling into the image smoothly. We start by training an autoencoder inspired by the paper Context Encoders by Deepak Pathek et al. We notice similar problems that the authors of the papers noticed, with the L2 loss function leading to blurry inpainting results.

However, instead of taking the approach outlined in their paper of using an adversarial network to provide an adversarial loss, we investigate alternate approaches to get rid of the blurring, such as using different loss functions that pay more attention to the structure of the image, such as SSIM, creating a custom loss based on image gradients to detect edge. Finally, we propose a workflow that uses Unsharp Masking and mean+median filtering to integrate the generated infilling into the image smoothly. Furthermore, we investigate the latent space of our encoder using principle component analysis, and find out that it meaningfully differs from the latent space of the ResNet encoder.

**Checkout the full write up and code of this project in the [notebook.ipynb](https://github.com/shahanneda/ImageInpainting/blob/main/notebook.ipynb)**


### References
https://www.cs.cmu.edu/~dpathak/context_encoder