import gradio as gr
from dependency_injector.providers import Configuration

from utils.cloud_storage.minio_storage import MinioStorage
from utils.url_handler import cloud_upload


_config = Configuration()
_config.minio.endpoint.from_env("MINIO_ENDPOINT")
_config.minio.access_key.from_env("MINIO_ACCESS_KEY")
_config.minio.secreet_key.from_env("MINIO_PRIVATE_KEY")
_config.minio.bucket_name.from_env("MINIO_BUCKET_NAME")
minio_storage = MinioStorage(
    _config.minio.endpoint(),
    _config.minio.access_key(),
    _config.minio.secreet_key(),
    bucket_name=_config.minio.bucket_name(),
    secure=False,
)


def upload_file(file):
    public_url, upload_result = cloud_upload(cloud_storage=minio_storage, local_path=file)
    print(public_url)


def welcome(name):
    return f"Welcome to Gradio, {name}!"


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Upload file to MinIO
    """
    )
    upload_button = gr.UploadButton()
    upload_button.upload(upload_file, upload_button)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
