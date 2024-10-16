import PIL.Image
import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from optimum.quanto import freeze, qfloat8, quantize
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


class Flux:
    def __init__(self, *args, **kwargs):
        dtype = torch.bfloat16

        # schnell is the distilled turbo model. For the CFG distilled model, use:
        # bfl_repo = "black-forest-labs/FLUX.1-dev"
        # revision = "refs/pr/3"
        #
        # The undistilled model that uses CFG ("pro") which can use negative prompts
        # was not released.
        bfl_repo = "black-forest-labs/FLUX.1-schnell"
        revision = "refs/pr/1"

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision
        )
        vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
        transformer = FluxTransformer2DModel.from_pretrained(
            bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision
        )

        # Experimental: Try this to load in 4-bit for <16GB cards.
        #
        # from optimum.quanto import qint4
        # quantize(transformer, weights=qint4, exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"])
        # freeze(transformer)
        quantize(transformer, weights=qfloat8)
        freeze(transformer)

        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        self._pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        self._pipe.text_encoder_2 = text_encoder_2
        self._pipe.transformer = transformer
        self._pipe.enable_model_cpu_offload()

    def run(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator_seed: int,
        *args,
        **kwargs
    ) -> PIL.Image.Image:
        generator = torch.Generator().manual_seed(generator_seed)
        image = self._pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        )[0][0]
        return image
