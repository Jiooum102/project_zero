from gradio_demo.getting_started.containers.app_container import AppContainer
from gradio_demo.getting_started.ui.ui import make_app_ui


if __name__ == "__main__":
    container = AppContainer()
    container.config.from_yaml(
        "/mnt/data/0.Workspace/0.SourceCode/project_zero/gradio_demo/getting_started/configs.yaml"
    )
    container.wire(modules=["gradio_demo.getting_started.ui.ui"])

    # Make ui
    ui = make_app_ui()
    # os.environ['GRADIO_TEMP_DIR'] = container.config.gradio.temp_dir()
    ui.launch(
        server_name=container.config.gradio.server_name(),
        server_port=container.config.gradio.server_port(),
        share=container.config.gradio.share(),
    )
