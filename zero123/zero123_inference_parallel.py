import argparse
import math
import time
from contextlib import nullcontext

import numpy as np
import torch
import transformers
from diffusers.models.autoencoder_kl import DiagonalGaussianDistribution
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
from transformers import AutoFeatureExtractor

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import (
    create_carvekit_interface,
    instantiate_from_config,
    load_and_preprocess,
)


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def preprocess_image(models, input_im, preprocess):
    """
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    """

    print("old input_im:", input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models["carvekit"], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(
        f"Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s."
    )
    print("new input_im:", lo(input_im))

    return input_im


@torch.no_grad()
def sample_model(
    input_im,
    model,
    sampler,
    precision,
    h,
    w,
    ddim_steps,
    n_samples,
    scale,
    ddim_eta,
    x,
    y,
    z,
):
    precision_scope = autocast if precision == "autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)

            T = torch.tensor(
                [
                    math.radians(x),
                    math.sin(math.radians(y)),
                    math.cos(math.radians(y)),
                    z,
                ]
            )
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond["c_crossattn"] = [c]
            cond["c_concat"] = [
                model.encode_first_stage((input_im.to(c.device)))
                .mode()
                .detach()
                .repeat(n_samples, 1, 1, 1)
            ]
            if scale != 1.0:
                uc = {}
                uc["c_concat"] = [
                    torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)
                ]
                uc["c_crossattn"] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=None,
            )
            print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main(
    models,
    device,
    output,
    x=0.0,
    y=0.0,
    z=0.0,
    raw_im=None,
    preprocess=True,
    scale=3.0,
    n_samples=1,
    ddim_steps=50,
    ddim_eta=1.0,
    precision="fp32",
    h=256,
    w=256,
):

    # safety_checker_input = models["clip_fe"](raw_im, return_tensors="pt").to(device)
    # (_, has_nsfw_concept) = models["nsfw"](
    #     images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values
    # )
    # print("has_nsfw_concept:", has_nsfw_concept)
    # if np.any(has_nsfw_concept):
    #     print("NSFW content detected.")
    #     return
    #
    # print("Safety check passed.")

    input_im = preprocess_image(models, raw_im, preprocess)

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])

    sampler = DDIMSampler(models["turncam"])
    x_samples_ddim = sample_model(
        input_im,
        models["turncam"],
        sampler,
        precision,
        h,
        w,
        ddim_steps,
        n_samples,
        scale,
        ddim_eta,
        x,
        y,
        z,
    )

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    print(output_ims)
    output_ims[0].save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image")
    parser.add_argument("--ckpt", type=str, default="weights/zero123-xl.ckpt")
    parser.add_argument(
        "--config", type=str, default="configs/sd-objaverse-finetune-c_concat-256.yaml"
    )
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    config = OmegaConf.load(args.config)
    raw_im = Image.open(args.input)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print("Instantiating LatentDiffusion...")
    models["turncam"] = load_model_from_config(config, args.ckpt, device=device)
    # print("Instantiating Carvekit HiInterface...")
    # models["carvekit"] = create_carvekit_interface()
    # print("Instantiating StableDiffusionSafetyChecker...")
    # models["nsfw"] = StableDiffusionSafetyChecker.from_pretrained(
    #     "CompVis/stable-diffusion-safety-checker"
    # ).to(device)
    # print("Instantiating AutoFeatureExtractor...")
    # models["clip_fe"] = AutoFeatureExtractor.from_pretrained(
    #     "CompVis/stable-diffusion-safety-checker"
    # )
    #
    # models["nsfw"].concept_embeds_weights *= 1.07
    # models["nsfw"].special_care_embeds_weights *= 1.07

    angles = [
        [16.22, 194.04],
        [16.22, 204.04],
        [16.22, 214.04],
        [16.22, 224.04],
        [16.22, 234.04],
        [16.22, 244.04],
        [16.22, 254.04],
        [16.22, 264.04],
        [16.22, 274.04],
        [16.22, 284.04],
        [16.22, 294.04],
        [16.22, 304.04],
        [16.22, 314.04],
    ]

    for i in range(len(angles)):
        print(f"Angles: ({angles[i][0]}, {angles[i][1]})")

        main(
            models=models,
            device=device,
            output=f"../data/0808_results/12/backpack_side2/zero123_{angles[i][0]}_{angles[i][1]}.png",
            x=angles[i][0],
            y=angles[i][1],
            z=0,
            raw_im=raw_im,
            preprocess=False,
        )
