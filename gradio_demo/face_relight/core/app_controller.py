import logging
import os.path
import uuid
from typing import Dict, Union

import numpy as np
from PIL.Image import Image

from gradio_demo.base.core import MinioWrapper, MongoClientWrapper
from gradio_demo.face_relight.core.ic_light_wrapper import ICLightWrapper
from gradio_demo.face_relight.core.models import ICLightInfo
from sdk.utils.fileio import save_image


class AppController:
    def __init__(
        self,
        minio_storage: MinioWrapper,
        mongo_db: MongoClientWrapper,
        ic_light: ICLightWrapper,
        config: Dict,
    ):
        self._minio_storage = minio_storage
        self._mongo_db = mongo_db
        self._ic_light = ic_light
        self._config = config
        self._temp_dir = config['temp_dir']
        self._logger = logging.getLogger()

    def save_image_to_minio(self, image: Union[str, np.ndarray, Image]) -> str:
        if image is None or image == "None":
            return str(image)

        image_type = type(image)
        if image_type in [np.ndarray, Image]:
            local_path = os.path.join(self._temp_dir, f"{uuid.uuid4()}.png")
            save_image(image, local_path)
            image_url = self._minio_storage.upload(local_path, remove_local=True)
            return image_url
        else:  # filepath
            if not os.path.isfile(image):
                self._logger.warning(f"File not found: {image}")
                return image
            image_url = self._minio_storage.upload(image)
            return image_url

    def _log_request(
        self,
        input_image,
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
        input_fg,
        matting,
        results,
    ):
        ic_light_info = ICLightInfo(
            input_image=self.save_image_to_minio(input_image),
            prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            cfg=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
            lowres_denoise=lowres_denoise,
            bg_source=bg_source,
            input_fg=self.save_image_to_minio(input_fg),
            matting=self.save_image_to_minio(matting),
            results=[self.save_image_to_minio(result) for result in results],
        )
        self._mongo_db.insert_one_request(data=ic_light_info.model_dump())

    def process_relight(
        self,
        input_image,
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
        input_fg, matting = self._ic_light.run_rmbg(input_image)
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
        self._log_request(
            input_image,
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
            input_fg,
            matting,
            results,
        )
        return input_fg, results
