## SADPM: Synchronous Image-label Diffusion with Anisotropic Noise for Medical Image Segmentation
This is the official implementation of the paper "Synchronous Image-label Diffusion with Anisotropic Noise for Medical Image Segmentation" for review purposes. This journal paper extends our previous MICCAI paper "Synchronous Image-Label Diffusion with Anisotropic Noise for Stroke Lesion Segmentation on Non-Contrast CT".


## Environment Setup
This implementation runs on:

Ubuntu `20.04.6 LTS`

Python `3.9`

PyTorch `2.0.1`

CUDA `11.7`

Additional dependencies are listed in `requirements.txt`.

The codebase is built upon the official repository of `denoising_diffusion_pytorch == 2.1.1`. Familiarity with DDPM code is recommended.

## Directory Structure


`./data`: Contains datasets. Organize your data according to the structure specified below.


`SADPM_standard.py` : Training implementation including dataloader, training loops, noise generation, etc.

❗ Important: Replace the original DDPM code denoising_diffusion_pytorch.py with this file and add "Unet2" to the package file `__init__.py`:

```python
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer,Unet2
```

`test_standard.py` : Main function for parameter configuration.
 
`./Results/Predict` : Directory for saving inference results.

`./Results/Testset` : Test data directory.

`./Results/TrainingResults` : Directory for training logs and checkpoints.

