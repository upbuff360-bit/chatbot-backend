import asyncio
import os
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = AsyncIOMotorClient(os.environ["MONGO_URI"])
    db = client[os.environ["MONGO_DB_NAME"]]

    agents = await db.agents.find({}, {"_id": 1, "name": 1}).to_list(100)

    agents_root = Path(os.environ.get("AGENTS_DATA_ROOT", "./data/agents"))
    disk_folders = [f.name for f in agents_root.iterdir() if f.is_dir()] if agents_root.exists() else []

    print("\n=== MongoDB Agent IDs ===")
    for a in agents:
        print(f"  {a['_id']}  |  {a['name']}")

    print("\n=== Folders on disk (data/agents/) ===")
    for f in sorted(disk_folders):
        print(f"  {f}")

    client.close()

asyncio.run(main())
