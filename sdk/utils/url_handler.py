from typing import Any

from sdk.cloud_storage.minio_storage import MinioStorage


def cloud_upload(cloud_storage: Any, local_path: str, *args, **kwargs) -> None:
    if isinstance(cloud_storage, MinioStorage):
        return cloud_storage.upload(local_path, *args, **kwargs)
    else:
        raise NotImplementedError(
            f"Cloud storage type {cloud_storage.__class__} not supported! Current available cloud storage types are {MinioStorage}"
        )
