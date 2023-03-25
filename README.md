# Denoisemoji

This is a command-line tool that allows you to create a denoised emoji from a PNG/WEBP/JPG image.

## Usage
The tool can be run from a command prompt and requires several arguments

```bash
python denoisemoji.py -i <input_image_path> [-o <output_image_path>] [-sd <path_to_diffusers_model>] [--no-upscale] [-n <number_steps>] [-t <take_every>] [-s <size>] [-d <device>] [-dt <dtype>]
```

### Required arguments
* `-i`: Path to the input PNG file

### Optional arguments
* `-o`: Path to the output file. If not specified, it will be saved in the same directory as the input image with `-denoised.gif` appended to the file name
* `-sd`: Path to the [diffusers model](https://github.com/huggingface/diffusers) to use. If not specified, a default model will be used
* `--no-up`: By default, the input image is scaled up using the [Real-ESRGAN model](https://github.com/xinntao/Real-ESRGAN) before denoising, which is useful with emojis, since they are very small, the images will be automatically resized to 512x512 for optimal denoising resolution. Use this flag to disable this feature.
* `-n`: Number of diffusion denoising steps. Default is 100
* `-t`: How many images to discard vs total number of images generated. Default is 3, which means that steps/3 images will be saved to the gif.
* `-s`: Size (height and width) of the output GIF in pixels (output will be a square GIF). Default is 64
* `-d`: Device to run inference on. Default is "cuda" if a GPU is available, otherwise "cpu"
* `-dt`: Data type to use as torch tensors when decoding images for the VAE. Default is fp32, as occasionally decoded images at fp16 result in black frames.


## Requirements
This tool requires:
* Python 3.8 or later
* [PyTorch](https://pytorch.org/)
* [diffusers](https://github.com/huggingface/diffusers)
* [realesrgan](https://github.com/xinntao/Real-ESRGAN)
* [tqdm] (https://pypi.org/project/tqdm/)
* [Pillow](https://pypi.org/project/Pillow/)

All dependencies can be installed via pip.
