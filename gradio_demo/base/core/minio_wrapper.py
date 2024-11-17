import os

from sdk.cloud_storage.minio_storage import MinioStorage
from sdk.utils.url_handler import cloud_upload


class MinioWrapper:
    def __init__(self, *args, **kwargs):
        self.__minio = MinioStorage(*args, **kwargs)

    def upload(self, file: str, remove_local: bool = False):
        """

        :param file: File to upload
        :param remove_local: If true, local file will be deleted after upload
        :return:
        """
        try:
            public_url, upload_result = cloud_upload(cloud_storage=self.__minio, local_path=file)
            print(f"File uploaded to: {public_url}")

            if remove_local:
                print(f"Removing: {file}")
                os.remove(file)

            return public_url
        except Exception as e:
            return str(e)
