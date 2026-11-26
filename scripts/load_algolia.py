# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Load the retail catalog into an Algolia index: settings, then one record per listing.
The API's search (``AlgoliaRetail``) reads the index this builds.

    python scripts/load_algolia.py --dry-run     # print the settings and the first record
    python scripts/load_algolia.py --export examples/retail/data/algolia-catalog.json
                                                 # write the records as a JSON array, to upload
                                                 # in the Algolia dashboard (no credentials)
    python scripts/load_algolia.py               # replace the index's records

Needs ``ALGOLIA_APP_ID``, ``ALGOLIA_INDEX``, and ``ALGOLIA_ADMIN_KEY`` (a write key; the
API itself only gets the search-only ``ALGOLIA_SEARCH_KEY``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from demo_common import load_demo_env  # noqa: E402
from demo_common.storefront_fixtures import load_catalog  # noqa: E402
from retail.api.algolia_retail import INDEX_SETTINGS, index_records  # noqa: E402
from retail.api.mock_retail import DATA_DIR  # noqa: E402


def wait_for(client: httpx.Client, base: str, response: httpx.Response) -> None:
    response.raise_for_status()
    task_id = response.json()["taskID"]
    for _ in range(60):
        status = client.get(f"{base}/task/{task_id}").json().get("status")
        if status == "published":
            return
        time.sleep(0.5)
    raise SystemExit(f"Algolia task {task_id} did not publish in 30 seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print, change nothing")
    parser.add_argument("--export", type=Path, help="write the records to this JSON file")
    args = parser.parse_args()

    _, listings, _ = load_catalog(DATA_DIR)
    records = index_records(listings.values())
    if args.export:
        args.export.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(records)} records to {args.export}")
        return 0
    if args.dry_run:
        print(json.dumps(INDEX_SETTINGS, indent=2))
        print(f"\n{len(records)} records; the first:\n{json.dumps(records[0], indent=2)}")
        return 0

    load_demo_env(DATA_DIR.parent)
    try:
        app_id, index, key = (
            os.environ["ALGOLIA_APP_ID"],
            os.environ["ALGOLIA_INDEX"],
            os.environ["ALGOLIA_ADMIN_KEY"],
        )
    except KeyError as missing:
        raise SystemExit(f"Set {missing.args[0]} (see .env.example)") from None

    base = f"https://{app_id}.algolia.net/1/indexes/{index}"
    headers = {"X-Algolia-Application-Id": app_id, "X-Algolia-API-Key": key}
    with httpx.Client(headers=headers, timeout=30.0) as client:
        wait_for(client, base, client.put(f"{base}/settings", json=INDEX_SETTINGS))
        wait_for(client, base, client.post(f"{base}/clear"))
        batch = {"requests": [{"action": "addObject", "body": r} for r in records]}
        wait_for(client, base, client.post(f"{base}/batch", json=batch))
        found = client.post(f"{base}/query", json={"query": "", "hitsPerPage": 0}).json()
    print(f"Loaded {len(records)} records into {index}; the index reports {found['nbHits']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
