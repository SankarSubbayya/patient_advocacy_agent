#!/usr/bin/env python3
"""
Check for label files in the SCIN GCS bucket.
"""

from google.cloud import storage

def check_for_labels():
    """Check GCS bucket for label files."""
    bucket_name = "dx-scin-public-data"

    print("Checking for label files in SCIN bucket...")
    print("="*60)

    try:
        # Connect anonymously
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)

        # Look for CSV/TSV files
        print("\nSearching for CSV/TSV files...")
        blobs = bucket.list_blobs()

        csv_files = []
        for blob in blobs:
            if blob.name.endswith(('.csv', '.tsv', '.txt')):
                csv_files.append(blob.name)
                print(f"  Found: {blob.name}")

                # If it looks like labels, show first few KB
                if any(word in blob.name.lower() for word in ['label', 'annotation', 'diagnosis', 'condition', 'class']):
                    print(f"    ^ This might contain labels!")

        if not csv_files:
            print("  No CSV/TSV files found in bucket")
        else:
            print(f"\nTotal files found: {len(csv_files)}")
            print("\nTo download a specific file, use:")
            print("  gsutil cp gs://dx-scin-public-data/<filename> .")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTry running:")
        print("  gcloud auth application-default login")

if __name__ == "__main__":
    check_for_labels()