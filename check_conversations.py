"""
check_conversations.py — checks if conversation agent_ids match MongoDB agent _ids
Run: python check_conversations.py
"""
import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = AsyncIOMotorClient(os.environ["MONGO_URI"])
    db = client[os.environ["MONGO_DB_NAME"]]

    agents = await db.agents.find({}, {"_id": 1, "name": 1}).to_list(100)
    agent_ids = {a["_id"] for a in agents}
    agent_map  = {a["_id"]: a["name"] for a in agents}

    convs = await db.conversations.find({}, {"_id": 1, "agent_id": 1, "title": 1}).to_list(500)

    print(f"\nTotal agents      : {len(agents)}")
    print(f"Total conversations: {len(convs)}")

    matched   = [c for c in convs if c["agent_id"] in agent_ids]
    unmatched = [c for c in convs if c["agent_id"] not in agent_ids]

    print(f"Matched convs     : {len(matched)}")
    print(f"Unmatched convs   : {len(unmatched)}")

    if unmatched:
        print("\n--- Unmatched conversations (orphaned) ---")
        for c in unmatched[:10]:
            print(f"  conv_id={c['_id'][:8]}  agent_id={c['agent_id'][:8]}  title={c['title'][:40]}")

    if matched:
        print("\n--- Matched conversations ---")
        for c in matched[:10]:
            print(f"  conv_id={c['_id'][:8]}  agent={agent_map.get(c['agent_id'], '?')[:20]}  title={c['title'][:40]}")

    client.close()

asyncio.run(main())
