import os
import uuid
from typing import Dict, Union

import gradio

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
        config: Dict,
    ):
        self.__minio_storage = minio_storage
        self.__flux = flux
        self.__mongo_db = mongo_db
        self.__config = config

        self.__session_infor: Dict[str, Dict[str, Union[FluxInput, FluxOutput]]] = {}

    def create_new_session(self):
        session_id = str(uuid.uuid4())
        self.__session_infor[session_id] = {
            'input': FluxInput(),
            'output': FluxOutput(),
        }
        return session_id

    def update_input_prompt(self, user_id: str, new_input_prompt: str):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"

        self.__session_infor[user_id]['input'].prompt = new_input_prompt
        return []

    def update_width(self, user_id: str, width: int):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]['input'].width = width
        return []

    def update_height(self, user_id: str, height: int):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]["input"].height = height
        return []

    def update_num_inference_steps(self, user_id: str, num_inference_steps: int):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]["input"].num_inference_steps = num_inference_steps
        return []

    def update_generator_seed(self, user_id: str, generator_seed: int):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]['input'].generator_seed = generator_seed
        return []

    def update_guidance_scale(self, user_id: str, guidance_scale: float):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]["input"].guidance_scale = guidance_scale
        return []

    def btn_start_demo_clicked(self):
        session_id = self.create_new_session()
        return session_id, gradio.Row(visible=True), gradio.Button(visible=False), gradio.Textbox(visible=True)

    def run(self, session_id: str):
        """

        :param session_id: User ID
        :return:
        """

        # Run flux
        model_input = self.__session_infor[session_id]['input']
        output_image = self.__flux.run(**model_input.model_dump())

        # Upload minio
        tmp_dir: str = self.__config["gradio"]["temp_dir"]
        output_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.png")
        output_image.save(output_path)

        output_url = self.__minio_storage.upload(output_path)

        output = FluxOutput(output_url=output_url, output_path=output_path)

        # Update db
        record_info = {
            "user_id": session_id,
            "input": model_input.model_dump(),
            "output": output.model_dump(exclude={'output_record_id'}),
        }
        request_id = self.__mongo_db.insert_one_request(record_info)

        output.output_record_id = request_id
        self.__session_infor[session_id]['output'] = output
        return output_path, output_url, request_id
