import logging
import os
import traceback
from typing import Union

import cv2
import numpy as np
import PIL.Image
from PIL.Image import Image


logger = logging.getLogger()


def save_image(image: Union[np.ndarray, Image], save_path: str):
    # Make save dir
    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    # Try save using PIL
    try:
        # Convert to PIL.Image
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)

        image.save(save_path)

    except Exception as e:
        logger.debug(f"Can not save image using PIL with error {e}, using OpenCV instead. \n{traceback.format_exc()}")
        cv2.imwrite(save_path, image)

    return save_path

    raise NotImplementedError(f"Not supported image type: {type(image)}")
