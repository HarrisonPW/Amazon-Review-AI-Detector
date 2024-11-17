import json
import os
from datetime import datetime
from confluent_kafka import Consumer, KafkaError, OFFSET_BEGINNING
from pymongo import MongoClient

mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://mongodb_container:27017/'))
db = mongo_client['spam_detection']
reviews_collection = db['reviews']

# Kafka Consumer Configuration
conf = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092'),
    'group.id': 'spam_detection_group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': True,
    'auto.commit.interval.ms': 1000,
    'session.timeout.ms': 30000,
    'max.poll.interval.ms': 300000
}


def reset_offset(consumer, partitions):
    for p in partitions:
        p.offset = OFFSET_BEGINNING
    consumer.assign(partitions)


consumer = Consumer(conf)
consumer.subscribe(['spam_detection_topic'], on_assign=reset_offset)


def store_in_mongodb(message):
    try:
        if isinstance(message.get('timestamp'), str):
            message['timestamp'] = datetime.fromisoformat(message['timestamp'])

        result = reviews_collection.insert_one(message)
        print(f"Stored message in MongoDB with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        print(f"Error storing in MongoDB: {e}")
        return None


def main():
    print("Starting Kafka consumer...")
    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print('Reached end of partition')
                else:
                    print(f'Error: {msg.error()}')
                continue

            try:
                message_value = json.loads(msg.value().decode('utf-8'))
                print(f"Received message: {message_value}")
                store_in_mongodb(message_value)
            except json.JSONDecodeError as e:
                print(f"Error decoding message: {e}")
            except Exception as e:
                print(f"Unexpected error processing message: {e}")

    except KeyboardInterrupt:
        print("Shutting down consumer...")
    finally:
        consumer.close()
        mongo_client.close()


if __name__ == "__main__":
    main()
