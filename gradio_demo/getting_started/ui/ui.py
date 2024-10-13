from cProfile import label

import gradio
from dependency_injector.wiring import Provide, inject
from streamlit.web.server import allow_cross_origin_requests

from gradio_demo.getting_started.containers.app_container import AppContainer
from gradio_demo.getting_started.core.controller import AppController
from gradio_demo.getting_started.models.inputs import FluxInput


@inject
def make_app_ui(
    app_controller: AppController = Provide[AppContainer.app_controller],
):
    with gradio.Blocks() as demo:
        with gradio.Row(variant='panel'):
            gradio.Markdown(
                """
                # Welcome to Gradio Demo
                """
            )
            btn_start_demo = gradio.Button("Start Demo")
            session_id = gradio.Textbox(label="Session ID", visible=False, interactive=False)
        with gradio.Row(show_progress=False, visible=False) as main_row:
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
                with gradio.Row():
                    btn_generate_images = gradio.Button("Generate images")
                    btn_clear = gradio.ClearButton()
                    btn_load_examples = gradio.Button("Load examples")

            with gradio.Column(show_progress=True):
                gradio.Markdown(
                    """
                    # Outputs
                    """
                )
                request_id = gradio.Textbox(label="Request ID")
                output_url = gradio.Textbox(label="Output url", interactive=False, visible=True)
                output_image = gradio.Image(type="filepath", format="png", show_download_button=True, interactive=False)

        with gradio.Column(show_progress=True):
            examples = gradio.Examples(
                examples=[app_controller.get_examples(limit=5)],
                inputs=[
                    request_id,
                    input_prompt,
                    output_url,
                    image_width,
                    image_height,
                    num_inference_step,
                    generator_seed,
                    guidance_scale,
                ],
                cache_examples=False,
                visible=False,
            )

        # Setup action listener
        btn_start_demo.click(
            app_controller.btn_start_demo_clicked, None, [session_id, main_row, btn_start_demo, session_id]
        )
        input_prompt.change(app_controller.update_input_prompt, [session_id, input_prompt])
        image_width.change(app_controller.update_width, [session_id, image_width])
        image_height.change(app_controller.update_height, [session_id, image_height])
        num_inference_step.change(app_controller.update_num_inference_steps, [session_id, num_inference_step])
        generator_seed.change(app_controller.update_generator_seed, [session_id, generator_seed])
        guidance_scale.change(app_controller.update_guidance_scale, [session_id, guidance_scale])
        request_id.change(app_controller.update_request_id, [session_id, request_id])
        output_url.change(app_controller.update_output_url, [session_id, output_url])
        output_image.change(app_controller.update_output_image, [session_id, output_image])

        examples.load_input_event.success(app_controller.load_image_url, [request_id], [output_image])

        btn_load_examples.click(app_controller.btn_load_examples_clicked, None, [examples.dataset])
        btn_clear.add(
            components=[input_prompt, image_width, image_height, num_inference_step, generator_seed, guidance_scale]
        )
        btn_generate_images.click(
            app_controller.run, [session_id], [output_image, output_url, request_id], concurrency_limit=1
        )

    return demo
