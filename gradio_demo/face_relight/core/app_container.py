from dependency_injector import containers, providers

from gradio_demo.base.core.minio_wrapper import MinioWrapper
from gradio_demo.base.core.mongo_client_wrapper import MongoClientWrapper
from gradio_demo.face_relight.core.app_controller import AppController
from gradio_demo.face_relight.core.ic_light_wrapper import ICLightWrapper


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    ic_light = providers.Singleton(ICLightWrapper, device_id=config.ic_light.device_id)

    minio_storage = providers.Factory(
        MinioWrapper,
        endpoint=config.minio_storage.endpoint,
        access_key=config.minio_storage.access_key,
        secret_key=config.minio_storage.secret_key,
        bucket_name=config.minio_storage.bucket_name,
        secure=config.minio_storage.secure,
    )
    mongo_db = providers.Singleton(
        MongoClientWrapper,
        endpoint=config.mongo_db.endpoint,
        username=config.mongo_db.username,
        password=config.mongo_db.password,
        database=config.mongo_db.database,
        users_collection=config.mongo_db.users_collection,
        requests_collection=config.mongo_db.requests_collection,
    )

    app_controller = providers.Singleton(
        AppController,
        minio_storage=minio_storage,
        mongo_db=mongo_db,
        config=config,
        ic_light=ic_light,
    )
