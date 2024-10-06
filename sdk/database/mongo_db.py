import urllib.parse

import pymongo


class MongoDB:
    def __init__(self, username: str, password: str, endpoint: str):
        mongodb_uri = f"mongodb://{username}:{urllib.parse.quote(password)}@{endpoint}"
        self._client = pymongo.MongoClient(mongodb_uri, authSource="admin", connect=True)

    def insert_one(self, database: str, collection: str, document: dict):
        collection = self._client.get_database(database).get_collection(collection)
        return collection.insert_one(document)

    def update_one(self, database: str, collection: str, document_id: str, document: dict):
        collection = self._client.get_database(database).get_collection(collection)
        collection.update_one(
            filter={"_id": document_id},
            update={"$set": document},
        )


if __name__ == '__main__':
    mongodb = MongoDB(username="admin", password="mongodb123", endpoint="localhost:27017")
    mongodb.insert_one(database="app", collection="request", document={"msg": "test"})
