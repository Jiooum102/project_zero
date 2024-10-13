import datetime

from gradio_demo.getting_started.models.user import User
from sdk.database.mongo_db import MongoDB


class MongoClientWrapper:
    def __init__(
        self,
        database: str,
        users_collection: str,
        requests_collection: str,
        username: str,
        password: str,
        endpoint: str,
    ):
        self._mongo_client = MongoDB(username=username, password=password, endpoint=endpoint)
        self._database = database
        self._users_collection = users_collection
        self._request_collection = requests_collection

    def insert_request(self, data, *args, **kwargs):
        doc = {k.label: v for k, v in data.items()}
        return self.insert_one_request(doc, *args, **kwargs)

    def insert_one_request(self, data: dict, *args, **kwargs):
        create_time = datetime.datetime.now()
        data.update({'create_time': create_time})
        result = self._mongo_client.insert_one(self._database, self._request_collection, data)
        inserted_id = str(result.inserted_id)
        return inserted_id

    def insert_email(self, email: str):
        query = {"email": email}
        query_result = self._mongo_client.find_one(self._database, self._users_collection, query)

        # Email already existed in db
        if query_result is not None:
            user = User(**query_result)
        else:
            # Create new user
            user = User(email=email)
            insert_result = self._mongo_client.insert_one(
                self._database, self._users_collection, document=user.model_dump()
            )
            print(f"Inserted new user with document id: {insert_result.inserted_id}")
        return user.user_id

    def get_latest_requests(self, limit: int = 50):
        return self._mongo_client.find(self._database, self._request_collection, query={}).sort('create_time', -1)[
            :limit
        ]
