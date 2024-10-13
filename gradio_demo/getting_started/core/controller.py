import os
import uuid
from typing import Dict, Union

import gradio

from gradio_demo.getting_started.core.flux_wrapper import FluxWrapper
from gradio_demo.getting_started.core.minio_wrapper import MinioWrapper
from gradio_demo.getting_started.core.mongo_client_wrapper import MongoClientWrapper
from gradio_demo.getting_started.models.inputs import FluxInput
from gradio_demo.getting_started.models.outputs import FluxOutput
from sdk.utils.download import download_file


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

    def update_output_url(self, user_id: str, output_url: str):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]["output"].output_url = output_url
        return []

    def update_output_image(self, user_id: str, output_path: str):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]["output"].output_path = output_path
        return []

    def update_request_id(self, user_id: str, request_id: str):
        assert user_id in self.__session_infor, f"Not found session information of user id: {user_id}"
        self.__session_infor[user_id]["output"].output_record_id = request_id
        return []

    def btn_start_demo_clicked(self):
        session_id = self.create_new_session()
        return session_id, gradio.Row(visible=True), gradio.Button(visible=False), gradio.Textbox(visible=True)

    def get_examples(self, limit: int = 5):
        records = self.__mongo_db.get_latest_requests(limit=limit)
        _examples = []
        for record in records:
            request_id = str(record["_id"])
            _input = FluxInput()
            _input.prompt = record['input']['prompt']
            _input.width = record['input']['width']
            _input.height = record['input']['height']
            _input.num_inference_steps = record['input']['num_inference_steps']
            _input.generator_seed = record['input']['generator_seed']
            _input.guidance_scale = record['input']['guidance_scale']
            _output = FluxOutput(**record["output"])
            _examples.append(
                [
                    request_id,
                    _input.prompt,
                    _output.output_url,
                    _input.width,
                    _input.height,
                    _input.num_inference_steps,
                    _input.generator_seed,
                    _input.guidance_scale,
                ]
            )
        return _examples

    def btn_load_examples_clicked(self):
        _examples = self.get_examples(limit=50)
        return gradio.Dataset(samples=_examples, visible=True)

    def load_image_url(self, request_id: str) -> str:
        _request_infor = self.__mongo_db.find_request(request_id=request_id)
        _output = FluxOutput(**_request_infor['output'])
        if not os.path.exists(_output.output_path):
            download_file(url=_output.output_url, save_path=_output.output_path)
        return _output.output_path

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
