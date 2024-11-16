from gradio_demo.face_relight.core.app_container import AppContainer
from gradio_demo.face_relight.core.ui import make_app_ui


if __name__ == "__main__":
    container = AppContainer()
    container.config.from_yaml("gradio_demo/face_relight/configs.yaml")
    container.wire(modules=["gradio_demo.face_relight.core.ui"])

    # Make ui
    ui = make_app_ui()

    ui.launch(
        server_name=container.config.gradio.server_name(),
        server_port=container.config.gradio.server_port(),
        share=container.config.gradio.share(),
    )
