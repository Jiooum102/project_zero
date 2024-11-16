import cv2
import numpy as np

from sdk.models.ic_light import ICLight


class ICLightWrapper:
    def __init__(self, device_id: int):
        self._device_id = device_id
        self._ic_light = ICLight(self._device_id)

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
        for _input in [input_fg, bg_source]:
            if isinstance(_input, str) and _input != 'None':
                _input = cv2.imread(_input)
                _input = cv2.cvtColor(_input, cv2.COLOR_BGR2RGB)

        results = self._ic_light.process(
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
        )
        return results

    def run_rmbg(self, img, sigma=0.0):
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._ic_light.run_rmbg(img, sigma)
