import asyncio
import os
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient

# Same env keys and defaults as backend/services/mongodb.py
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://admin:elsee@cluster0.uuiqxb1.mongodb.net/",
).strip()
TEST_DB = os.getenv("MONGODB_DB_NAME", "seefore").strip()


def utc_now():
    return datetime.now(timezone.utc)


async def mongo_test():
    client = AsyncIOMotorClient(MONGODB_URI)
    try:
        await client.admin.command("ping")
        print("✅ Connected to MongoDB!")

        dbs = await client.list_database_names()
        print("Databases:", dbs)

        db = client[TEST_DB]
        collection = db["test_collection"]

        dummy_doc = {"message": "Hello MongoDB!", "created_at": utc_now()}
        result = await collection.insert_one(dummy_doc)
        print(f"Inserted dummy document with _id: {result.inserted_id}")

        doc = await collection.find_one({"_id": result.inserted_id})
        print("Retrieved document:", doc)

    except Exception as e:
        print("❌ MongoDB operation failed:", e)
    finally:
        client.close()


asyncio.run(mongo_test())
