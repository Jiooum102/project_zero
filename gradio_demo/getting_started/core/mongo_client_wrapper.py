import datetime
import uuid

from gradio_client.documentation import document

from sdk.database.mongo_db import MongoDB


class MongoClientWrapper:
    def __init__(
        self,
        database: str,
        collection: str,
        username: str,
        password: str,
        endpoint: str,
    ):
        self._mongo_client = MongoDB(username=username, password=password, endpoint=endpoint)
        self._database = database
        self._collection = collection

    def create(self, data, *args, **kwargs):
        doc = {k.label: v for k, v in data.items()}
        create_time = datetime.datetime.now()
        doc.update({'create_time': create_time})

        result = self._mongo_client.insert_one(self._database, self._collection, doc)
        inserted_id = str(result.inserted_id)
        print(f"Created request_id: {inserted_id}!")
        return inserted_id
