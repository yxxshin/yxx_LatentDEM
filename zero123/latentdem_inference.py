import argparse
import math
import random
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

from ldm.models.diffusion.ddim import LatentDEMSampler
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


def sample_model(
    input_imgs,
    init_poses,
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
    skip_Mstep,
):
    precision_scope = autocast if precision == "autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():

            img_conds = []
            phis = []

            for i, input_im in enumerate(input_imgs):
                img_cond_dict = {}

                c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
                img_cond_dict["c_original"] = c
                img_cond_dict["c_crossattn"] = None
                img_cond_dict["c_concat"] = [
                    model.encode_first_stage((input_im.to(c.device)))
                    .mode()
                    .detach()
                    .repeat(n_samples, 1, 1, 1)
                ]

                # Only use x, y, z as phi_1 and fix it (not optimized).
                if i == 0:
                    phi = torch.tensor([x, y, z], requires_grad=False)

                else:
                    if init_poses is not None:
                        random_x = x - init_poses[i - 1][0]
                        random_y = y - init_poses[i - 1][1]
                        random_z = z - init_poses[i - 1][2]

                    else:
                        # Following zero123 demo settings
                        random.seed(1234)
                        random_x = random.uniform(-90.0, 90.0)
                        random_y = random.uniform(-180.0, 180.0)
                        # random_z = random.uniform(-0.5, 0.5)
                        random_z = 0

                    if skip_Mstep:
                        # Phi is not optimizd
                        phi = torch.tensor(
                            [random_x, random_y, random_z], requires_grad=False
                        )

                    else:
                        # Phi optimized
                        phi = torch.tensor(
                            [random_x, random_y, random_z], requires_grad=True
                        )

                    print(f"\n*** Initialized phis *** : {phi}\n")

                img_conds.append(img_cond_dict)

                phis.append(phi)

            shape = [4, h // 8, w // 8]

            assert len(img_conds) == len(phis)
            img_num = len(img_conds)

            z_samples, phis = sampler.sample(
                S=ddim_steps,
                batch_size=n_samples,
                shape=shape,
                img_num=img_num,
                img_conds=img_conds,
                phis=phis,
                verbose=False,
                unconditional_guidance_scale=scale,
                eta=ddim_eta,
                x_T=None,
                skip_Mstep=skip_Mstep,
            )

            print(f"phis: {phis}")
            x_samples = model.decode_first_stage(z_samples)
            return torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main(
    models,
    device,
    output,
    x=0.0,
    y=0.0,
    z=0.0,
    img_paths=None,
    init_poses=None,
    preprocess=True,
    skip_Mstep=False,
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

    input_imgs = []

    print(f"Input images: {img_paths}")

    for path in img_paths:
        input_im = Image.open(path)

        input_im = preprocess_image(models, input_im, preprocess)
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])

        input_imgs.append(input_im)

    sampler = LatentDEMSampler(models["turncam"])

    x_samples_ddim = sample_model(
        input_imgs,
        init_poses,
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
        skip_Mstep,
    )

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.detach().cpu().numpy(), "c h w -> h w c")
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    output_ims[0].save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=float, required=True)
    parser.add_argument("-y", type=float, required=True)
    parser.add_argument("-z", type=float, required=True)
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input images (Text File)"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output file location"
    )
    parser.add_argument(
        "--init_pose", "-p", type=str, help="Initial pose of images (phi2, phi3, ...)"
    )
    parser.add_argument(
        "--use_gt",
        action="store_true",
        help="If on, skip M step and use initial pose",
    )
    parser.add_argument("--ckpt", type=str, default="weights/zero123-xl.ckpt")
    parser.add_argument(
        "--config", type=str, default="configs/sd-objaverse-finetune-c_concat-256.yaml"
    )
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    config = OmegaConf.load(args.config)

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

    with open(args.input, "r") as file:
        paths = file.readlines()
        paths = [path.strip() for path in paths]

    if args.init_pose is not None:
        poses = []

        with open(args.init_pose, "r") as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()

            if line:
                pose = line.split(",")
                pose = [float(value) for value in pose]
                poses.append(pose)

    else:
        poses = None

    if poses == None and args.use_gt:
        raise ValueError("Initial pose needed for use_gt mode")

    main(
        models=models,
        device=device,
        output=args.output,
        x=args.x,
        y=args.y,
        z=args.z,
        img_paths=paths,
        init_poses=poses,
        preprocess=False,
        skip_Mstep=args.use_gt,
    )
