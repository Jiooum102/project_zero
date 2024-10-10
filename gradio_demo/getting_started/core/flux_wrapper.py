import gc

import PIL.Image
import torch

from sdk.models.flux import Flux


class FluxWrapper:
    def __init__(self, *args, **kwargs):
        self._flux: Flux = None

    def run(self, n_items: int = 1, *args, **kwargs) -> PIL.Image.Image:
        if self._flux is None:
            self._flux = Flux()

        output_images = []
        for i in range(n_items):
            img = self._flux.run(*args, **kwargs)
            output_images.append(img)

        del self._flux
        self._flux = None

        torch.cuda.empty_cache()
        gc.collect()
        return output_images[0]
