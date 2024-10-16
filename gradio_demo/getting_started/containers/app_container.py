from dependency_injector import containers, providers

from gradio_demo.getting_started.core.controller import AppController
from gradio_demo.getting_started.core.flux_wrapper import FluxWrapper
from gradio_demo.getting_started.core.minio_wrapper import MinioWrapper
from gradio_demo.getting_started.core.mongo_client_wrapper import MongoClientWrapper


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

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
    flux = providers.Singleton(FluxWrapper)

    app_controller = providers.Singleton(
        AppController,
        minio_storage=minio_storage,
        flux=flux,
        mongo_db=mongo_db,
        config=config,
    )
