from typing import Dict

from gradio_demo.base.core import MinioWrapper, MongoClientWrapper
from gradio_demo.face_relight.core.ic_light_wrapper import ICLightWrapper


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

    def upload_file(self, local_path: str) -> str:
        return self._minio_storage.upload(local_path)

    def process_relight(
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
        input_fg, matting = self._ic_light.run_rmbg(input_fg)
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
        return input_fg, results
