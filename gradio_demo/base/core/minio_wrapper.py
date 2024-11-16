from sdk.cloud_storage.minio_storage import MinioStorage
from sdk.utils.url_handler import cloud_upload


class MinioWrapper:
    def __init__(self, *args, **kwargs):
        self.__minio = MinioStorage(*args, **kwargs)

    def upload(self, file):
        try:
            public_url, upload_result = cloud_upload(cloud_storage=self.__minio, local_path=file)
            print(f"File uploaded to: {public_url}")
            return public_url
        except Exception as e:
            return str(e)
