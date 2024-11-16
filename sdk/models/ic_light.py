import math
import os
from enum import Enum

import numpy as np
import safetensors.torch as sf
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image
from torch.hub import download_url_to_file
from transformers import CLIPTextModel, CLIPTokenizer

from sdk.models.briarmbg import BriaRMBG


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


class ICLight:
    def __init__(self, device_id: int):
        self._device = torch.device(device_id)

        # 'stablediffusionapi/realistic-vision-v51'
        # 'runwayml/stable-diffusion-v1-5'
        sd15_name = 'stablediffusionapi/realistic-vision-v51'
        self._tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
        self._text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
        self._vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
        self._unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
        self._rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

        # Change UNet
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                8,
                self._unet.conv_in.out_channels,
                self._unet.conv_in.kernel_size,
                self._unet.conv_in.stride,
                self._unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self._unet.conv_in.weight)
            new_conv_in.bias = self._unet.conv_in.bias
            self._unet.conv_in = new_conv_in

        unet_original_forward = self._unet.forward

        def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
            c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
            new_sample = torch.cat([sample, c_concat], dim=1)
            kwargs['cross_attention_kwargs'] = {}
            return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

        self._unet.forward = hooked_unet_forward

        # Load

        model_path = './models/iclight_sd15_fc.safetensors'
        if not os.path.exists(model_path):
            download_url_to_file(
                url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors',
                dst=model_path,
            )

        sd_offset = sf.load_file(model_path)
        sd_origin = self._unet.state_dict()
        keys = sd_origin.keys()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        self._unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged, keys

        self._text_encoder = self._text_encoder.to(device=self._device, dtype=torch.float16)
        self._vae = self._vae.to(device=self._device, dtype=torch.bfloat16)
        self._unet = self._unet.to(device=self._device, dtype=torch.float16)
        self._rmbg = self._rmbg.to(device=self._device, dtype=torch.float32)

        # SDP
        self._unet.set_attn_processor(AttnProcessor2_0())
        self._vae.set_attn_processor(AttnProcessor2_0())

        # Samplers

        self._ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self._euler_a_scheduler = EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, steps_offset=1
        )

        self._dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1,
        )

        # Pipelines

        self._t2i_pipe = StableDiffusionPipeline(
            vae=self._vae,
            text_encoder=self._text_encoder,
            tokenizer=self._tokenizer,
            unet=self._unet,
            scheduler=self._dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        )

        self._i2i_pipe = StableDiffusionImg2ImgPipeline(
            vae=self._vae,
            text_encoder=self._text_encoder,
            tokenizer=self._tokenizer,
            unet=self._unet,
            scheduler=self._dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        )

    @staticmethod
    def _pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    @torch.inference_mode()
    def encode_prompt_inner(self, txt: str):
        max_length = self._tokenizer.model_max_length
        chunk_length = self._tokenizer.model_max_length - 2
        id_start = self._tokenizer.bos_token_id
        id_end = self._tokenizer.eos_token_id
        id_pad = id_end

        tokens = self._tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i : i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [self._pad(ck, id_pad, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=self._device, dtype=torch.int64)
        conds = self._text_encoder(token_ids).last_hidden_state

        return conds

    @torch.inference_mode()
    def encode_prompt_pair(self, positive_prompt, negative_prompt):
        c = self.encode_prompt_inner(positive_prompt)
        uc = self.encode_prompt_inner(negative_prompt)

        c_len = float(len(c))
        uc_len = float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat = int(math.ceil(max_count / c_len))
        uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)

        return c, uc

    @torch.inference_mode()
    def run_rmbg(self, img, sigma=0.0):
        H, W, C = img.shape
        assert C == 3
        k = (256.0 / float(H * W)) ** 0.5
        feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
        feed = numpy2pytorch([feed]).to(device=self._device, dtype=torch.float32)
        alpha = self._rmbg(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
        result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
        return result.clip(0, 255).astype(np.uint8), alpha

    @torch.inference_mode()
    def process(
        self,
        input_fg,
        prompt,
        image_width,
        image_height,
        num_samples,
        seed,
        steps,
        a_prompt,
        n_prompt,
        cfg,
        highres_scale,
        highres_denoise,
        lowres_denoise,
        bg_source,
    ):
        bg_source = BGSource(bg_source)
        input_bg = None

        if bg_source == BGSource.NONE:
            pass
        elif bg_source == BGSource.LEFT:
            gradient = np.linspace(255, 0, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.RIGHT:
            gradient = np.linspace(0, 255, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.TOP:
            gradient = np.linspace(255, 0, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.BOTTOM:
            gradient = np.linspace(0, 255, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        else:
            raise 'Wrong initial latent!'

        rng = torch.Generator(device=self._device).manual_seed(int(seed))

        fg = resize_and_center_crop(input_fg, image_width, image_height)

        concat_conds = numpy2pytorch([fg]).to(device=self._vae.device, dtype=self._vae.dtype)
        concat_conds = self._vae.encode(concat_conds).latent_dist.mode() * self._vae.config.scaling_factor

        conds, unconds = self.encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

        if input_bg is None:
            latents = (
                self._t2i_pipe(
                    prompt_embeds=conds,
                    negative_prompt_embeds=unconds,
                    width=image_width,
                    height=image_height,
                    num_inference_steps=steps,
                    num_images_per_prompt=num_samples,
                    generator=rng,
                    output_type='latent',
                    guidance_scale=cfg,
                    cross_attention_kwargs={'concat_conds': concat_conds},
                ).images.to(self._vae.dtype)
                / self._vae.config.scaling_factor
            )
        else:
            bg = resize_and_center_crop(input_bg, image_width, image_height)
            bg_latent = numpy2pytorch([bg]).to(device=self._vae.device, dtype=self._vae.dtype)
            bg_latent = self._vae.encode(bg_latent).latent_dist.mode() * self._vae.config.scaling_factor
            latents = (
                self._i2i_pipe(
                    image=bg_latent,
                    strength=lowres_denoise,
                    prompt_embeds=conds,
                    negative_prompt_embeds=unconds,
                    width=image_width,
                    height=image_height,
                    num_inference_steps=int(round(steps / lowres_denoise)),
                    num_images_per_prompt=num_samples,
                    generator=rng,
                    output_type='latent',
                    guidance_scale=cfg,
                    cross_attention_kwargs={'concat_conds': concat_conds},
                ).images.to(self._vae.dtype)
                / self._vae.config.scaling_factor
            )

        pixels = self._vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)
        pixels = [
            resize_without_crop(
                image=p,
                target_width=int(round(image_width * highres_scale / 64.0) * 64),
                target_height=int(round(image_height * highres_scale / 64.0) * 64),
            )
            for p in pixels
        ]

        pixels = numpy2pytorch(pixels).to(device=self._vae.device, dtype=self._vae.dtype)
        latents = self._vae.encode(pixels).latent_dist.mode() * self._vae.config.scaling_factor
        latents = latents.to(device=self._unet.device, dtype=self._unet.dtype)

        image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

        fg = resize_and_center_crop(input_fg, image_width, image_height)
        concat_conds = numpy2pytorch([fg]).to(device=self._vae.device, dtype=self._vae.dtype)
        concat_conds = self._vae.encode(concat_conds).latent_dist.mode() * self._vae.config.scaling_factor

        latents = (
            self._i2i_pipe(
                image=latents,
                strength=highres_denoise,
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=int(round(steps / highres_denoise)),
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type='latent',
                guidance_scale=cfg,
                cross_attention_kwargs={'concat_conds': concat_conds},
            ).images.to(self._vae.dtype)
            / self._vae.config.scaling_factor
        )

        pixels = self._vae.decode(latents).sample

        return pytorch2numpy(pixels)
