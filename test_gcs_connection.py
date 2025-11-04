#!/usr/bin/env python3
"""
Test Google Cloud Storage connection for public SCIN dataset.
"""

def test_anonymous_access():
    """Test anonymous access to public bucket."""
    print("Testing anonymous access to SCIN public bucket...\n")
    
    try:
        from google.cloud import storage
    except ImportError:
        print("✗ google-cloud-storage not installed")
        print("\nInstall with:")
        print("  uv pip install google-cloud-storage")
        return False
    
    bucket_name = "dx-scin-public-data"
    
    # Try 1: Anonymous client
    print("Method 1: Anonymous Client")
    print("-" * 60)
    try:
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        
        # Try to list some files
        blobs = list(bucket.list_blobs(prefix="dataset/", max_results=5))
        print(f"✓ SUCCESS! Found {len(blobs)} files in bucket")
        print(f"  Sample files:")
        for blob in blobs[:3]:
            print(f"    - {blob.name}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    # Try 2: No-auth client (older method)
    print("Method 2: No-Auth Client")
    print("-" * 60)
    try:
        from google.auth import credentials
        
        class AnonymousCredentials(credentials.Credentials):
            """Anonymous credentials for public buckets."""
            def refresh(self, request):
                pass
        
        client = storage.Client(
            project=None,
            credentials=AnonymousCredentials()
        )
        bucket = client.bucket(bucket_name)
        
        blobs = list(bucket.list_blobs(prefix="dataset/", max_results=5))
        print(f"✓ SUCCESS! Found {len(blobs)} files")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    # Try 3: Default credentials (requires gcloud auth)
    print("Method 3: Default Credentials (requires gcloud auth)")
    print("-" * 60)
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        blobs = list(bucket.list_blobs(prefix="dataset/", max_results=5))
        print(f"✓ SUCCESS! Found {len(blobs)} files")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        print("\nTo use this method, authenticate with:")
        print("  gcloud auth application-default login")
        return False

if __name__ == "__main__":
    import sys
    success = test_anonymous_access()
    
    if success:
        print("\n" + "="*60)
        print("✓ CONNECTION SUCCESSFUL!")
        print("="*60)
        print("\nYou can now run:")
        print("  uv run python download_scin_gcs.py")
    else:
        print("\n" + "="*60)
        print("✗ CONNECTION FAILED")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Install google-cloud-storage:")
        print("   uv pip install google-cloud-storage")
        print("\n2. For public access, anonymous should work")
        print("3. If not, install gcloud CLI:")
        print("   https://cloud.google.com/sdk/docs/install")
        print("   Then run: gcloud auth application-default login")
    
    sys.exit(0 if success else 1)


