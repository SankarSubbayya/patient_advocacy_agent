#!/usr/bin/env python3
"""Quick test of GCS access."""

print("Testing GCS access...", flush=True)

from google.cloud import storage

try:
    print("Creating anonymous client...", flush=True)
    client = storage.Client.create_anonymous_client()

    print("Accessing bucket...", flush=True)
    bucket = client.bucket("dx-scin-public-data")

    print("Listing first 5 blobs...", flush=True)
    count = 0
    for blob in bucket.list_blobs(prefix="dataset/images/"):
        print(f"  Found: {blob.name}", flush=True)
        count += 1
        if count >= 5:
            break

    print(f"\n✓ GCS access working! Found {count} images", flush=True)

except Exception as e:
    print(f"\n✗ Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
