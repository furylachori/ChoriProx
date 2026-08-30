#!/usr/bin/env python3
import os
import json
import shutil
from datetime import datetime

GOLDEN_DIR = os.path.dirname(__file__)
SCHEMA_PATH = os.path.join(GOLDEN_DIR, "../schemas/golden_response.schema.json")


def update_golden_response(test_name, new_response, approved=False):
    if not approved:
        print(f"[PRE-COMMIT] Would update golden response for '{test_name}'")
        print(f"  New response: {json.dumps(new_response, indent=2)}")
        print("  Set approved=True to commit changes.")
        return False

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    golden_file = os.path.join(GOLDEN_DIR, f"{test_name}.golden.json")

    entry = {
        "test_name": test_name,
        "model": "qwen",
        "temperature": 0,
        "seed": 42,
        "request": {"messages": [], "tools": []},
        "expected_response": new_response,
        "update_timestamp": timestamp,
        "update_mechanism": "update_golden.py",
    }

    with open(golden_file, "w") as f:
        json.dump(entry, f, indent=2)

    print(f"Updated: {golden_file}")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: update_golden.py <test_name> <response_json> [--approve]")
        sys.exit(1)

    test_name = sys.argv[1]
    response = json.loads(sys.argv[2])
    approved = "--approve" in sys.argv

    update_golden_response(test_name, response, approved)
