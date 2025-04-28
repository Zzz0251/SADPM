## SADPM: Synchronous Image-label Diffusion with Anisotropic Noise for Medical Image Segmentation
This is the official implementation of the paper "Synchronous Image-label Diffusion with Anisotropic Noise for Medical Image Segmentation" for reviewing. This journal paper is the extend of previous MICCAI paper "Synchronous Image-Label Diffusion with Anisotropic Noise for Stroke Lesion Segmentation on Non-Contrast CT".


## Setup the environment
This code is implemented in Ubuntu system `20.04.6 LTS`, with `Python 3.9`, `PyTorch 2.0.1` and `CUDA 11.7`, and other detailed packages are shown in `requiremnts.txt`

Code is builted upon the official repository of `denoising_diffusion_pytorch == 2.1.1`, so please make sure you are familar with the code of DDPM

## Directory description


`./data`: datasets used, you should organaize your data in the following structure and format


`SADPM_standard.py` : training file including dataloader, training, noise generate etc. 

❗ ❗ ❗ Use this code to **REPLACE** the original DDPM code `denoising_diffusion_pytorch.py` and add "Unet2" in the package file `__init__.py` 

```python
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer,**Unet2**
```

`test_standard.py` : main function to set parameters

`./Results/Predict` : save inference results

`./Results/Testset` : data for testing

`./Results/TrainingResults` : save trainig log and  

## complete code coming soon

The complete code will be released in the near future.
