import gradio
from dependency_injector.providers import Configuration
from dependency_injector.wiring import Provide, inject

from gradio_demo.getting_started.containers.app_container import AppContainer
from gradio_demo.getting_started.core.flux_wrapper import FluxWrapper
from gradio_demo.getting_started.core.minio_wrapper import MinioWrapper
from gradio_demo.getting_started.core.mongo_client_wrapper import MongoClientWrapper
from gradio_demo.getting_started.models.inputs import FluxInput


@inject
def make_app_ui(
    minio_storage: MinioWrapper = Provide[AppContainer.minio_storage],
    flux: FluxWrapper = Provide[AppContainer.flux],
    mongo_db: MongoClientWrapper = Provide[AppContainer.mongo_db],
    config: Configuration = Provide[AppContainer.config],
):
    with gradio.Blocks() as demo:
        with gradio.Row(show_progress=False):
            with gradio.Column(show_progress=False):
                gradio.Markdown(
                    """
                    # Inputs
                    """
                )
                input_prompt = gradio.Textbox(label="Input prompt", interactive=True)
                image_width = gradio.Slider(
                    label="Image width",
                    minimum=FluxInput.__MIN_SIZE__,
                    maximum=FluxInput.__MAX_SIZE__,
                    value=FluxInput.__DEFAULT_SIZE__,
                    step=8,
                    interactive=True,
                )
                image_height = gradio.Slider(
                    label="Image height",
                    minimum=FluxInput.__MIN_SIZE__,
                    maximum=FluxInput.__MAX_SIZE__,
                    value=FluxInput.__DEFAULT_SIZE__,
                    step=8,
                    interactive=True,
                )
                num_inference_step = gradio.Slider(
                    label="Number of inference steps",
                    minimum=FluxInput.__MIN_NUM_INFERENCE_STEPS__,
                    maximum=FluxInput.__MAX_NUM_INFERENCE_STEPS__,
                    value=FluxInput.__DEFAULT_NUM_INFERENCE_STEPS__,
                    step=1,
                    interactive=True,
                )
                generator_seed = gradio.Number(
                    label="Generator seed", value=FluxInput.__DEFAULT_GENERATOR_SEED__, interactive=True
                )
                guidance_scale = gradio.Slider(
                    label="Guidance scale",
                    minimum=FluxInput.__MIN_GUIDANCE_SCALE__,
                    maximum=FluxInput.__MAX_GUIDANCE_SCALE__,
                    value=FluxInput.__DEFAULT_GUIDANCE_SCALE__,
                    interactive=True,
                )
                n_items = gradio.Number(value=1, visible=False)

                btn_generate_images = gradio.Button("Generate images")

            with gradio.Column(show_progress=True):
                gradio.Markdown(
                    """
                    # Outputs
                    """
                )
                request_id = gradio.Textbox(label="Request ID")
                output_url = gradio.Textbox(label="Output url", interactive=False, visible=True)
                output_image = gradio.Image(type="filepath", format="png", show_download_button=True, interactive=False)
            btn_generate_images.click(
                flux.run,
                [n_items, input_prompt, image_width, image_height, num_inference_step, guidance_scale, generator_seed],
                [output_image],
            )
            output_image.change(minio_storage.upload, [output_image], [output_url])
            output_url.change(
                mongo_db.create,
                inputs={
                    input_prompt,
                    image_width,
                    image_height,
                    num_inference_step,
                    guidance_scale,
                    generator_seed,
                    output_url,
                },
                outputs=[request_id],
            )
    return demo
