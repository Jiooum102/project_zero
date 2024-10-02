import mimetypes
import os
import uuid

from dependency_injector.providers import AbstractSingleton, Configuration
from minio import Minio

from utils.cloud_storage.base_cloud_storage import AbstractCloudStorage


class MinioStorage(AbstractCloudStorage):
    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, secure: bool, bucket_name: str = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._endpoint = endpoint
        self._bucket_name = str(uuid.uuid4()) if bucket_name is None else bucket_name
        self._secure = secure

        if self._secure:
            self._endpoint_url = "https://" + self._endpoint
        else:
            self._endpoint_url = "http://" + self._endpoint

        self._client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def upload(self, local_path: str, object_name: str = None, metadata: dict = None, *args, **kwargs):
        if object_name is None:
            file_ext = os.path.splitext(os.path.basename(local_path))[-1]
            object_name = "default/" + str(uuid.uuid4()) + file_ext
        kwargs.update({'content_type': mimetypes.guess_type(object_name)[0]})
        upload_result = self._client.fput_object(
            self._bucket_name, object_name, local_path, metadata=metadata, *args, **kwargs
        )
        public_url = f"{self._endpoint_url}/{self._bucket_name}/{object_name}"
        return public_url, upload_result
