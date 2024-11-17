from typing import List

from pydantic import BaseModel


class ICLightInfo(BaseModel):
    input_image: str
    prompt: str
    image_width: int
    image_height: int
    num_samples: int
    seed: int
    steps: int
    a_prompt: str
    n_prompt: str
    cfg: int
    highres_scale: float
    highres_denoise: float
    lowres_denoise: float
    bg_source: str
    input_fg: str
    matting: str
    results: List[str]
