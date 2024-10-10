import copy
import os
import uuid
from typing import Any, Dict, Union

import cv2
from dependency_injector.providers import Configuration

from gradio_demo.getting_started.core.flux_wrapper import FluxWrapper
from gradio_demo.getting_started.core.minio_wrapper import MinioWrapper
from gradio_demo.getting_started.core.mongo_client_wrapper import MongoClientWrapper
from gradio_demo.getting_started.models.inputs import FluxInput
from gradio_demo.getting_started.models.outputs import FluxOutput


class AppController:
    def __init__(
        self,
        minio_storage: MinioWrapper,
        flux: FluxWrapper,
        mongo_db: MongoClientWrapper,
        config: Configuration,
    ):
        self.__minio_storage = minio_storage
        self.__flux = flux
        self.__mongo_db = mongo_db
        self.__config = config

        self.__session_infor: Dict[str, Dict[str, Union[FluxInput, FluxOutput]]] = {}

    def create_temp_user_id(self):
        temp_user_id = str(uuid.uuid4())
        self.__session_infor[temp_user_id] = {"input": FluxInput(), "output": FluxOutput().model_fields_set}
        return temp_user_id

    def update_user_id(self, temp_user_id: str, user_id: str):
        if user_id not in self.__session_infor:
            self.__session_infor[user_id] = {}

        self.__session_infor[user_id].update(self.__session_infor[temp_user_id])
        del self.__session_infor[temp_user_id]

    def update_input_prompt(self, user_id: str, new_input_prompt: str):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"

        self.__session_infor[user_id]['input'].prompt = new_input_prompt

    def update_width(self, user_id: str, width: int):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]['input'].width = width

    def update_height(self, user_id: str, height: int):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]["input"].height = height

    def update_num_inference_steps(self, user_id: str, num_inference_steps: int):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]["input"].num_inference_steps = num_inference_steps

    def update_generator_seed(self, user_id: str, generator_seed: int):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]['input'].generator_seed = generator_seed

    def update_guidance_scale(self, user_id: str, guidance_scale: float):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]["input"].guidance_scale = guidance_scale

    def run(self, user_id: str):
        # Run flux
        model_input = self.__session_infor[user_id]['input']
        output_image = self.__flux.run(**model_input.model_dump())

        # Upload minio
        tmp_dir: str = self.__config["gradio"]["temp_dir"]
        output_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.png")
        output_image.save(output_path)

        output_url = self.__minio_storage.upload(output_path)

        output = FluxOutput(output_url=output_url, output_path=output_path)

        # Update db
        record_info = {
            "user_id": user_id,
            "input": model_input.model_dump(),
            "output": output.model_dump(exclude=set('output_record_id')),
        }
        request_id = self.__mongo_db.insert_one(record_info)

        output.output_record_id = request_id
        self.__session_infor[user_id]['output'] = output
        return output_path, output_url, request_id
