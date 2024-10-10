import gradio
from dependency_injector.providers import Configuration
from dependency_injector.wiring import Provide, inject

from gradio_demo.getting_started.containers.app_container import AppContainer
from gradio_demo.getting_started.core.controller import AppController
from gradio_demo.getting_started.models.inputs import FluxInput


@inject
def make_app_ui(
    app_controller: AppController = Provide[AppContainer.app_controller],
):
    with gradio.Blocks() as demo:
        with gradio.Row():
            temp_user_id = app_controller.create_temp_user_id()
            user_id = gradio.Textbox(label="User ID", value=temp_user_id, visible=True, interactive=False)

        with gradio.Row(show_progress=False):
            with gradio.Column(show_progress=False):
                gradio.Markdown(
                    """
                    # Inputs
                    """
                )
                input_prompt = gradio.Textbox(label="Input prompt", interactive=True, value="")
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

            # Setup action listener
            input_prompt.change(app_controller.update_input_prompt, [user_id, input_prompt], None)
            image_width.change(app_controller.update_width, [user_id, image_width], None)
            image_height.change(app_controller.update_height, [user_id, image_height], None)
            num_inference_step.change(app_controller.update_num_inference_steps, [user_id, num_inference_step], None)
            generator_seed.change(app_controller.update_generator_seed, [user_id, generator_seed], None)
            guidance_scale.change(app_controller.update_guidance_scale, [user_id, guidance_scale], None)

            btn_generate_images.click(
                app_controller.run, [user_id], [output_image, output_url, request_id], concurrency_limit=1
            )
    return demo
