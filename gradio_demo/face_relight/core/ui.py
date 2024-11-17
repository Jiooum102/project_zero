import gradio
from dependency_injector.wiring import Provide, inject

from gradio_demo.face_relight.core import db_examples
from gradio_demo.face_relight.core.app_container import AppContainer
from gradio_demo.face_relight.core.app_controller import AppController
from sdk.models.ic_light import BGSource


quick_prompts = [
    'sunshine from window',
    'neon light, city',
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm',
]
quick_prompts = [[x] for x in quick_prompts]

quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]


def process_relight(*args, **kwargs):
    return None


@inject
def make_app_ui(
    app_controller: AppController = Provide[AppContainer.app_controller],
):
    demo = gradio.Blocks().queue()
    with demo:
        with gradio.Row():
            gradio.Markdown("## IC-Light (Relighting with Foreground Condition)")
        with gradio.Row():
            with gradio.Column():
                with gradio.Row():
                    input_fg = gradio.Image(sources=['upload'], type="filepath", label="Image", height=480)
                    input_fg_url = gradio.Textbox(label="input_fg_url", visible=False, interactive=False)
                    output_bg = gradio.Image(type="filepath", label="Preprocessed Foreground", height=480, format='png')
                    output_bg_url = gradio.Textbox(label="output_bg_url", visible=False, interactive=False)

                prompt = gradio.Textbox(label="Prompt")
                bg_source = gradio.Radio(
                    choices=[e.value for e in BGSource],
                    value=BGSource.NONE.value,
                    label="Lighting Preference (Initial Latent)",
                    type='value',
                )
                example_quick_subjects = gradio.Dataset(
                    samples=quick_subjects, label='Subject Quick List', samples_per_page=1000, components=[prompt]
                )
                example_quick_prompts = gradio.Dataset(
                    samples=quick_prompts, label='Lighting Quick List', samples_per_page=1000, components=[prompt]
                )
                relight_button = gradio.Button(value="Relight")

                with gradio.Group():
                    with gradio.Row():
                        num_samples = gradio.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                        seed = gradio.Number(label="Seed", value=12345, precision=0)

                    with gradio.Row():
                        image_width = gradio.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                        image_height = gradio.Slider(
                            label="Image Height", minimum=256, maximum=1024, value=640, step=64
                        )

                with gradio.Accordion("Advanced options", open=False):
                    steps = gradio.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                    cfg = gradio.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=2, step=0.01)
                    lowres_denoise = gradio.Slider(
                        label="Lowres Denoise (for initial latent)", minimum=0.1, maximum=1.0, value=0.9, step=0.01
                    )
                    highres_scale = gradio.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                    highres_denoise = gradio.Slider(
                        label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.01
                    )
                    a_prompt = gradio.Textbox(label="Added Prompt", value='best quality')
                    n_prompt = gradio.Textbox(
                        label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality'
                    )
            with gradio.Column():
                result_gallery = gradio.Gallery(height=832, object_fit='contain', label='Outputs')
        with gradio.Row():
            dummy_image_for_outputs = gradio.Image(visible=False, label='Result')
        gradio.Examples(
            fn=lambda *args: ([args[-1]], None),
            examples=db_examples.foreground_conditioned_examples,
            inputs=[input_fg, prompt, bg_source, image_width, image_height, seed, dummy_image_for_outputs],
            outputs=[result_gallery, output_bg],
            run_on_click=True,
            examples_per_page=1024,
        )

        # Setup action listener
        input_fg.change(fn=app_controller.save_image_to_minio, inputs=[input_fg], outputs=[input_fg_url])
        output_bg.change(fn=app_controller.save_image_to_minio, inputs=[output_bg], outputs=[output_bg_url])

        ips = [
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
        ]
        relight_button.click(fn=app_controller.process_relight, inputs=ips, outputs=[output_bg, result_gallery])
        example_quick_prompts.click(
            lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]),
            inputs=[example_quick_prompts, prompt],
            outputs=prompt,
            show_progress=False,
            queue=False,
        )
        example_quick_subjects.click(
            lambda x: x[0], inputs=example_quick_subjects, outputs=prompt, show_progress=False, queue=False
        )

    return demo
