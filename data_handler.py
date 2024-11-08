

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import time
import json
from bson import ObjectId
import uuid


def connect_to_mongo_with_retry(uri, retries=5, delay=5):
    for attempt in range(retries):
        try:
            client = MongoClient(uri)
            client.admin.command('ping')
            print("Connected to MongoDB")
            return client
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    raise Exception("Could not connect to MongoDB after multiple attempts.")


def load_data_to_mongodb():
    print("Starting data loading process...")
    mongo_uri = "mongodb://mongodb:27017"
    client = connect_to_mongo_with_retry(mongo_uri)
    db = client['MyDB']
    collection = db['Cell_Phones_and_Accessories']

    if collection.count_documents({}) > 0:
        print("Collection already contains data. Skipping data load.")
        return

    try:
        with open('Cell_Phones_and_Accessories.json', 'r') as file:
            documents = json.load(file)
            for document in documents:
                if '_id' in document and isinstance(document['_id'], dict) and '$oid' in document['_id']:
                    document['_id'] = ObjectId(document['_id']['$oid'])

                try:
                    collection.insert_one(document)
                except DuplicateKeyError:
                    document['_id'] = ObjectId()
                    document['unique_id'] = str(uuid.uuid4())
                    collection.insert_one(document)

            print(f"Successfully loaded {collection.count_documents({})} documents into MongoDB")

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


if __name__ == "__main__":
    load_data_to_mongodb()
