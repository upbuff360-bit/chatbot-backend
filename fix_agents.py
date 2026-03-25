"""
fix_agents.py — fixes two problems:
  1. Removes duplicate agents from MongoDB (migration ran twice)
  2. Renames disk folders from old IDs to new MongoDB IDs

Run from chatbot/ project root:
    python fix_agents.py
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

STATE_PATH   = Path(os.environ.get("STATE_JSON_PATH", "./persist/admin/state.json"))
AGENTS_ROOT  = Path(os.environ.get("AGENTS_DATA_ROOT", "./data/agents"))


async def main():
    client = AsyncIOMotorClient(os.environ["MONGO_URI"])
    db = client[os.environ["MONGO_DB_NAME"]]

    # ── Step 1: Load old state.json to get old_id → name mapping ─────────────
    print("\n=== Step 1: Reading state.json ===")
    with STATE_PATH.open("r", encoding="utf-8") as f:
        state = json.load(f)

    # old_id → agent_name
    old_id_to_name = {a["id"]: a["name"] for a in state["agents"]}
    print(f"Found {len(old_id_to_name)} agents in state.json")
    for old_id, name in old_id_to_name.items():
        print(f"  {old_id[:8]}...  |  {name}")

    # ── Step 2: Remove duplicate MongoDB agents ───────────────────────────────
    print("\n=== Step 2: Removing duplicate MongoDB agents ===")

    all_agents = await db.agents.find({}, {"_id": 1, "name": 1, "created_at": 1}).to_list(1000)

    # Group by name, keep the one with the earliest created_at
    from collections import defaultdict
    by_name = defaultdict(list)
    for a in all_agents:
        by_name[a["name"]].append(a)

    keep_ids   = set()
    delete_ids = set()

    for name, group in by_name.items():
        # Sort by created_at, keep first
        group.sort(key=lambda x: x.get("created_at", ""))
        keep_ids.add(group[0]["_id"])
        for dup in group[1:]:
            delete_ids.add(dup["_id"])

    if delete_ids:
        result = await db.agents.delete_many({"_id": {"$in": list(delete_ids)}})
        # Also delete their documents and conversations
        await db.documents.delete_many({"agent_id": {"$in": list(delete_ids)}})
        await db.conversations.delete_many({"agent_id": {"$in": list(delete_ids)}})
        print(f"Deleted {result.deleted_count} duplicate agents")
    else:
        print("No duplicates found")

    # ── Step 3: Build name → new MongoDB ID mapping ───────────────────────────
    print("\n=== Step 3: Building name → new MongoDB ID mapping ===")

    remaining = await db.agents.find(
        {"_id": {"$in": list(keep_ids)}},
        {"_id": 1, "name": 1}
    ).to_list(100)

    name_to_new_id = {a["name"]: a["_id"] for a in remaining}
    for name, new_id in name_to_new_id.items():
        print(f"  {name}  →  {new_id[:8]}...")

    # ── Step 4: Rename disk folders old_id → new_id ───────────────────────────
    print("\n=== Step 4: Renaming disk folders ===")

    # name → old_id (from state.json)
    name_to_old_id = {v: k for k, v in old_id_to_name.items()}

    moved   = 0
    skipped = 0

    for name, new_id in name_to_new_id.items():
        old_id    = name_to_old_id.get(name)
        old_path  = AGENTS_ROOT / old_id if old_id else None
        new_path  = AGENTS_ROOT / new_id

        if new_path.exists():
            print(f"  ALREADY OK  '{name}'  ({new_id[:8]}...)")
            skipped += 1
            continue

        if old_path and old_path.exists():
            shutil.copytree(str(old_path), str(new_path))
            print(f"  MOVED  '{name}'  {old_id[:8]}... → {new_id[:8]}...")
            moved += 1
        else:
            # Create empty dirs so indexing doesn't fail
            for sub in ["pdfs", "websites", "text_snippets", "qa"]:
                (new_path / sub).mkdir(parents=True, exist_ok=True)
            print(f"  CREATED empty dirs  '{name}'  ({new_id[:8]}...)")
            skipped += 1

    print(f"\nFolders moved: {moved}  |  Already OK / created: {skipped}")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_agents = await db.agents.find({}, {"_id": 1, "name": 1}).to_list(100)
    print(f"\n=== Done ===")
    print(f"MongoDB now has {len(final_agents)} agents (no duplicates)")
    print("Now restart the backend to re-index all agents.")

    client.close()


asyncio.run(main())
