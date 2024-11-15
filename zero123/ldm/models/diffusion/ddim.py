"""SAMPLING ONLY."""

import copy
import math

import numpy as np
import torch
from lovely_numpy.repr_plt import sample
from tqdm import tqdm

from ldm.models.diffusion.sampling_util import norm_thresholding
from ldm.modules.diffusionmodules.util import (
    extract_into_tensor,
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def to(self, device):
        """Same as to in torch module
        Don't really underestand why this isn't a module in the first place"""
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                new_v = getattr(self, k).to(device)
                setattr(self, k, new_v)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(
        self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True
    ):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,  # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        dynamic_threshold=None,
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                    )

            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for DDIM sampling is {size}, eta {eta}")

        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            dynamic_threshold=dynamic_threshold,
        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        t_start=-1,
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (
                int(
                    min(timesteps / self.ddim_timesteps.shape[0], 1)
                    * self.ddim_timesteps.shape[0]
                )
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        timesteps = timesteps[:t_start]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = (
            reversed(range(0, timesteps))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(
                    x0, ts
                )  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                dynamic_threshold=dynamic_threshold,
            )
            img, pred_x0 = outs
            if callback:
                img = callback(i, img, pred_x0)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
    ):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [
                            torch.cat([unconditional_conditioning[k][i], c[k][i]])
                            for i in range(len(c[k]))
                        ]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(
                self.model, e_t, x, t, c, **corrector_kwargs
            )

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            pred_x0 = norm_thresholding(pred_x0, dynamic_threshold)

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(
        self,
        x0,
        c,
        t_enc,
        use_original_steps=False,
        return_intermediates=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        num_reference_steps = (
            self.ddpm_num_timesteps
            if use_original_steps
            else self.ddim_timesteps.shape[0]
        )

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc="Encoding Image"):
            t = torch.full(
                (x0.shape[0],), i, device=self.model.device, dtype=torch.long
            )
            if unconditional_guidance_scale == 1.0:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(
                        torch.cat((x_next, x_next)),
                        torch.cat((t, t)),
                        torch.cat((unconditional_conditioning, c)),
                    ),
                    2,
                )
                noise_pred = e_t_uncond + unconditional_guidance_scale * (
                    noise_pred - e_t_uncond
                )

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = (
                alphas_next[i].sqrt()
                * ((1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt())
                * noise_pred
            )
            x_next = xt_weighted + weighted_noise_pred
            if (
                return_intermediates
                and i % (num_steps // return_intermediates) == 0
                and i < num_steps - 1
            ):
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)

        out = {"x_encoded": x_next, "intermediate_steps": inter_steps}
        if return_intermediates:
            out.update({"intermediates": intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    @torch.no_grad()
    def decode(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
    ):

        timesteps = (
            np.arange(self.ddpm_num_timesteps)
            if use_original_steps
            else self.ddim_timesteps
        )
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full(
                (x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long
            )
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        return x_dec


class LatentDEMSampler(DDIMSampler):
    def update_conds(self, img_conds, phis, scale, n_samples, shape, skip_Mstep):
        assert len(img_conds) == len(phis)

        conds = []
        ucs = []

        for i in range(len(img_conds)):
            conds_dict = {}
            conds_dict["c_concat"] = img_conds[i]["c_concat"]

            c = img_conds[i]["c_original"]

            x, y, z = phis[i]

            phi = torch.stack(
                [
                    torch.deg2rad(x),
                    torch.sin(torch.deg2rad(y)),
                    torch.cos(torch.deg2rad(y)),
                    z,
                ]
            )

            phi = phi[None, None, :].repeat(n_samples, 1, 1).to(c.device)

            c = torch.cat([c, phi], dim=-1)
            c = self.model.cc_projection(c)

            if skip_Mstep:
                c = c.detach()

            conds_dict["c_crossattn"] = [c]

            if scale != 1.0:
                uc = {}
                uc["c_concat"] = [
                    torch.zeros(n_samples, shape[1], shape[2], shape[3]).to(c.device)
                ]
                uc["c_crossattn"] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            conds.append(conds_dict)
            ucs.append(uc)

        return conds, ucs

    def sample(
        self,
        S,
        batch_size,
        shape,
        img_num=2,
        img_conds=None,
        phis=None,
        eta=0.0,
        temperature=1.0,
        noise_dropout=0.0,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        skip_Mstep=False,
        sample_from=None,
        Estep_scheduling=None,
        Mstep_scheduling=None,
        lr=1,
        lr_decay=False,
    ):

        device = self.model.betas.device

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for LatentDEM sampling is {size}, eta {eta}")

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        # FIXME: Consider nu_t (standard deviataion of time-dependent model errors)
        nus = torch.zeros_like(self.model.betas)

        print(f"Running LatentDEM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="LatentDEM Sampler", total=total_steps)

        # Initialize z_t(y_i, phi_i)
        z_list = []
        z_list_next = []

        if x_T is None:
            for j in range(img_num):
                z = torch.randn(size, device=device)
                z_list.append(z)

        else:
            z = x_T
            z_list = [z] * img_num

        # Initialize z_t(y_1, phi_1, ..., y_n, phi_n)
        if x_T is None:
            z_total = torch.randn(size, device=device)
        else:
            z_total = x_T

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            if not skip_Mstep:
                for img in range(1, img_num):
                    phis[img] = phis[img].clone().detach().requires_grad_(True)

            # Update self.conds according to phi updates
            conds, ucs = self.update_conds(
                img_conds=img_conds,
                phis=phis,
                scale=unconditional_guidance_scale,
                n_samples=batch_size,
                shape=size,
                skip_Mstep=skip_Mstep,
            )

            z_list_next = []

            assert len(z_list) == len(conds) == len(ucs) == img_num

            # Sample z_{t-1}(y_i, phi_i)
            for j in range(img_num):
                z_t = z_list[j]

                if sample_from == "start" and j != 0:
                    # For every timestep, sample from T (T -> t)

                    # For seed fix, uncomment the following line
                    # torch.manual_seed(123)

                    z_t = torch.randn(size, device=device)

                    for i_sfs, step_sfs in enumerate(iterator):

                        if i_sfs == i + 1:
                            break

                        index_sfs = total_steps - i_sfs - 1
                        print(f"index_sfs: {index_sfs}")
                        ts_sfs = torch.full(
                            (batch_size,), step_sfs, device=device, dtype=torch.long
                        )

                        if index_sfs == index:
                            print(f"-> z_list updated")
                            z_list[j] = z_t

                        if skip_Mstep:
                            with torch.no_grad():
                                z_t, _ = self.p_sample_ddim(
                                    z=z_t,
                                    c=conds[j],
                                    t=ts_sfs,
                                    index=index_sfs,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=ucs[j],
                                )
                        else:
                            z_t, _ = self.p_sample_ddim(
                                z=z_t,
                                c=conds[j],
                                t=ts_sfs,
                                index=index_sfs,
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=ucs[j],
                            )

                    z_t_next = z_t

                else:
                    if skip_Mstep:
                        with torch.no_grad():
                            z_t_next, _ = self.p_sample_ddim(
                                z=z_t,
                                c=conds[j],
                                t=ts,
                                index=index,
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=ucs[j],
                            )
                    else:
                        z_t_next, _ = self.p_sample_ddim(
                            z=z_t,
                            c=conds[j],
                            t=ts,
                            index=index,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=ucs[j],
                        )

                    torch.cuda.empty_cache()

                assert (
                    z_t_next.requires_grad == conds[j]["c_crossattn"][0].requires_grad
                )

                z_list_next.append(z_t_next)

            # E-step: z_{t-1}(y_1, phi_1, y_2, phi_2, ..., y_n, phi_n)
            z_total = self.E_step(
                z_total=z_total,
                index=index,
                nus=nus,
                z_list=z_list,
                z_list_next=z_list_next,
                img_num=img_num,
                phis=phis,
                Estep_scheduling=Estep_scheduling,
            )

            # M-step
            # Optimize phis, except phi_1

            if skip_Mstep:
                print(f"\npred_phi2: {phis}")

            else:
                phis = self.M_step(
                    phis=phis,
                    img_num=img_num,
                    z_total=z_total,
                    z_list_next=z_list_next,
                    index=index,
                    Mstep_scheduling=Mstep_scheduling,
                    lr=lr,
                    lr_decay=lr_decay,
                )

            if sample_from == "start":
                z_list_next[0] = z_total

            elif sample_from == "middle":
                z_list_next[0] = z_total
                z_list_next[1] = z_total

            elif sample_from == "prev":
                z_list_next[0] = z_total

            z_list = copy.copy(z_list_next)

        return z_total, phis

    def p_sample_ddim(
        self,
        z,
        c,
        t,
        index,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        repeat_noise=False,
        temperature=1.0,
        noise_dropout=0.0,
    ):
        """
        Sample z_{t-1}(y_i, phi_i) using Zero123, DDIM based

        """
        b, *_, device = *z.shape, z.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.model.apply_model(z, t, c)
        else:
            z_in = torch.cat([z] * 2)
            t_in = torch.cat([t] * 2)

            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [
                            torch.cat([unconditional_conditioning[k][i], c[k][i]])
                            for i in range(len(c[k]))
                        ]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(z_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        # Select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # Current prediction for z_0
        pred_z0 = (z - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # Direction pointing to z_t
        dir_zt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(z.shape, device, repeat_noise) * temperature

        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        z_prev = a_prev.sqrt() * pred_z0 + dir_zt + noise

        return z_prev, pred_z0

    def E_step(
        self, z_total, index, nus, z_list, z_list_next, img_num, phis, Estep_scheduling
    ):
        beta_t = torch.full(
            (z_total.shape[0], 1, 1, 1), self.betas[index], device=z_total.device
        )
        alpha_t = 1 - beta_t

        if Estep_scheduling == "linear":
            gamma_n = index / 100
            delta_n = 1 - gamma_n

        elif Estep_scheduling == "fixed":
            gamma_n = 0.5
            delta_n = 0.5

        elif Estep_scheduling == "curve":
            curve_1 = 1 - math.cos(math.radians(phis[0][0])) * math.cos(
                math.radians(phis[0][1])
            )
            curve_2 = 1 - math.cos(math.radians(phis[1][0])) * math.cos(
                math.radians(phis[1][1])
            )
            gamma_n = (curve_1) / (curve_1 + curve_2)
            delta_n = 1 - gamma_n

        else:
            print("Shouldn't come here")
            gamma_n = 0
            delta_n = 0

        # TODO: Using nus
        # nu_t = nus[index]
        # gamma_n = beta_t / (img_num * beta_t + nu_t**2)
        # delta_n = (beta_t + nu_t**2) / (img_num * beta_t + nu_t**2)

        print(f"gamma_n: {gamma_n}")

        z_weighted_sum = 0
        for i in range(len(z_list)):
            if i == 0:
                z_weighted_sum += delta_n * z_list[i]
            else:
                z_weighted_sum += gamma_n * z_list[i]

        z_weighted_sum_next = 0
        for i in range(len(z_list_next)):
            if i == 0:
                z_weighted_sum_next += delta_n * z_list_next[i]
            else:
                z_weighted_sum_next += gamma_n * z_list_next[i]

        z_total = (
            (1 / alpha_t.sqrt()) * z_total
            + z_weighted_sum_next
            - (1 / alpha_t.sqrt()) * z_weighted_sum
        )

        noise = noise_like(z_total.shape, z_total.device, False)
        # z_total = z_total + noise * beta_t / 2
        # z_total = (z_total + noise * beta_t / 2) * alpha_t.sqrt()

        return z_total

    def M_step(
        self,
        phis,
        img_num,
        z_total,
        z_list_next,
        index,
        Mstep_scheduling,
        lr,
        lr_decay,
    ):

        assert len(phis) == img_num

        # TODO: Lambda, Delta Scheduling

        if Mstep_scheduling == "linear":
            lambda_coef = index / 10
            delta_coef = 5

        elif Mstep_scheduling == "fixed":
            lambda_coef = 5
            delta_coef = 5

        else:
            print("Shouldn't come here")
            lambda_coef = 0
            delta_coef = 0

        phis_next = []
        phis_next.append(phis[0])

        z_list_next[0] = z_list_next[0].detach()

        for i in range(1, img_num):

            # phi_1 is fixed and not optimized

            # Calculating phi_i^{(t-1)}
            loss1 = torch.sum((z_list_next[i] - z_total) ** 2)
            loss2 = torch.sum((z_list_next[i] - z_list_next[0]) ** 2)

            loss = lambda_coef * loss1 + delta_coef * loss2

            phi_grad = torch.autograd.grad(
                outputs=loss,
                inputs=phis[i],
            )

            torch.cuda.empty_cache()

            # FIXME: Currently z is not updated
            phi_grad[0][2] = 0

            if lr_decay:
                lr = lr ** ((50 - index) / 50)

            phi_next = phis[i] - phi_grad[0] * lr

            phis_next.append(phi_next.detach())

            print(f"Loss(Latent):{loss2}, phi2_pred:{phi_next.detach().numpy()}\n")

        return phis_next
