from pathlib import Path
from PIL import Image

try:
    from diffusers.models.autoencoder_kl import AutoencoderKL
except Exception:
    from diffusers.models.vae import AutoencoderKL

from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
import numpy as np
import torch
from typing import Union
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.utils import RealESRGANer
from argparse import ArgumentParser as APP


def parse_args():
    args = APP(
        "Denoisemoji",
        description="Turn your emojis into .gif s. To generate a gif, please provide a path to a png image, e.g -i ~/smiley.webp",
    )
    args.add_argument("-i", "--emoji-path", type=Path, required=True)
    args.add_argument("-o", "--output-path", default=None, type=Path)
    args.add_argument(
        "-sd",
        "--diffusers-model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Diffusers model from remote repo or local path to instantiate the stable diffusion autoencoder.",
    )
    args.add_argument(
        "--no-up",
        "--no-upscale",
        dest="no_upscale",
        action="store_true",
        help="Do not scale up image by 2x using realesrgan. By default, an upscaler will be used.",
    )
    args.add_argument(
        "-n",
        "--steps",
        type=int,
        default=100,
        help="Number of diffusion denoising steps.",
    )
    args.add_argument(
        "-t",
        "--take-every",
        type=int,
        default=3,
        help="How many images to discard vs total number of images generated (will be equal to '3' by default, which means it will save steps/3 images to the gif).",
    )
    args.add_argument(
        "-s",
        "--size",
        type=int,
        default=64,
        help="Size (height and width) of the GIF in pixels (output will be a square GIF)",
    )
    args.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    args.add_argument(
        "-dt",
        "--dtype",
        type=str,
        choices=["fp32", "fp16"],
        default="fp32",
        help="Data type to use as torch tensors when decoding images for the vae, default is fp32, as occasionally decoded images at fp16 result in black frames.",
    )
    return args.parse_args()


args = parse_args()

moji_path: Path = args.emoji_path
moji_save_path: Path = (
    args.output_path
    if args.output_path
    else moji_path.with_name(moji_path.stem + "-denoised.gif")
)
diffusers_path: str = args.diffusers_model
upscale_input_emoji: bool = not args.no_upscale
denoise_steps: int = args.steps
take_image_every: int = args.take_every
output_gif_size: int = args.size
device: torch.device = torch.device(args.device)
dtype: torch.dtype = torch.float32 if args.dtype == "fp32" else torch.float16


class Upscaler:
    def __init__(
        self,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ) -> None:
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        self.upscaler = RealESRGANer(
            model_path=model_path,
            scale=2,
            model=self.model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )

    @torch.inference_mode()
    def upscale(
        self,
        img: Union[np.ndarray, Image.Image],
    ) -> Union[np.ndarray, Image.Image]:
        img: np.ndarray = np.asarray(img).copy()
        img = img[:, :, ::-1]
        img: np.ndarray = self.upscaler.enhance(img, outscale=2)[0]
        return Image.fromarray(img[:, :, ::-1])


im = Image.open(moji_path.as_posix()).convert("RGB")
if upscale_input_emoji:
    s = Upscaler()
    up = s.upscale(im)
    s.model.cpu()
    s.upscaler.model.cpu()

up = up.resize((512, 512), resample=Image.LANCZOS)

noise_scheduler: DPMSolverMultistepScheduler = (
    DPMSolverMultistepScheduler.from_pretrained(diffusers_path, subfolder="scheduler")
)


def prepare_image(
    image,
    width=512,
    height=512,
    batch_size=1,
    num_images_per_prompt=1,
    device="cuda",
    dtype=torch.float32,
):
    if isinstance(image, Image.Image):
        image = [image]

    if isinstance(image[0], Image.Image):
        image = [
            np.array(i.resize((width, height), resample=Image.LANCZOS))[None, :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)

    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    return (image * 2) - 1


vae: AutoencoderKL = (
    AutoencoderKL.from_pretrained(diffusers_path, subfolder="vae").to(device).to(dtype)
)

vae.enable_slicing()
try:
    vae.enable_tiling()
except Exception as e:
    print(f"Enabling tiling for vae decode failed, (which is fine)", flush=True)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def decode_latents(latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat32
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return [
        i.resize((output_gif_size, output_gif_size), Image.LANCZOS)
        for i in numpy_to_pil(image)
    ]


def make_gif(all_images):
    chosen_images = []
    for idx, image in enumerate(all_images):
        if idx % take_image_every == 0:
            chosen_images.append(image)
    frame_one: Image.Image = chosen_images[0]
    frame_one.save(
        moji_save_path,
        format="GIF",
        append_images=chosen_images[1:],
        save_all=True,
        duration=5,
        loop=0,
        optimize=True,
    )


all_images = []

prep = prepare_image(up).to(device).to(dtype)


with torch.no_grad():
    latents = vae.encode(prep).latent_dist.sample()
    latents = latents * 0.18215
    noise_scheduler.set_timesteps(denoise_steps, device=device)

    for x in tqdm(range(0, denoise_steps, 1), total=denoise_steps, desc="denoising"):

        noise = torch.randn_like(latents.to(dtype), device=device)

        noise = noise.contiguous()

        bsz = latents.shape[0]
        # Sample timestep for each image
        timesteps = torch.tensor(
            (x,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        all_images.extend(decode_latents(noisy_latents.to(device).to(dtype)))
        latents = noisy_latents

make_gif(list(reversed(all_images)))
