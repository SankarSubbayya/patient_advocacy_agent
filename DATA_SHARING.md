# Data Sharing Guide for Your Group

## Quick Summary

Your project has **~19GB of data and models** that need to be shared:
- **Data (SCIN)**: 12GB (images + metadata)
- **Models**: 6.9GB (trained embedder and indices)

---

## Method 1: Shared Network Drive (Recommended for Local Teams)

### Setup (Do this once as admin)

```bash
# Create shared directory on your server/NAS
sudo mkdir -p /mnt/shared_projects/scin_advocacy
sudo chown sankar:team /mnt/shared_projects/scin_advocacy
sudo chmod g+rx /mnt/shared_projects/scin_advocacy

# Copy your data
sudo cp -r /home/sankar/data/scin /mnt/shared_projects/scin_advocacy/
sudo cp -r /home/sankar/models /mnt/shared_projects/scin_advocacy/

# Set group permissions
sudo chmod -R g+rx /mnt/shared_projects/scin_advocacy/*
```

### For Team Members

```bash
# 1. Clone the project
git clone <repo-url>
cd patient_advocacy_agent

# 2. Create symlinks to shared data
mkdir -p data models
ln -s /mnt/shared_projects/scin_advocacy/scin data/scin
ln -s /mnt/shared_projects/scin_advocacy/models/* models/

# 3. Verify
ls -la data/scin/
ls -la models/embedder/
```

### Add to `.bashrc` for easy updates

```bash
# Add this to ~/.bashrc
alias sync_scin_data='rsync -avz --delete /mnt/shared_projects/scin_advocacy/scin ~/patient_advocacy_agent/data/'
alias sync_models='rsync -avz --delete /mnt/shared_projects/scin_advocacy/models/* ~/patient_advocacy_agent/models/'
```

---

## Method 2: Git LFS (Recommended for Version Control)

### Setup (Do this once)

```bash
# Install Git LFS
apt-get install git-lfs
cd ~/patient_advocacy_agent
git lfs install

# Track large files
git lfs track "data/scin/**"
git lfs track "models/**"
git add .gitattributes

# Commit and push
git add data/ models/
git commit -m "Add data and models with Git LFS"
git push origin main
```

### For Team Members

```bash
# Clone with LFS
git clone <repo-url>
git lfs pull

# Or if already cloned
cd patient_advocacy_agent
git lfs pull
```

---

## Method 3: Cloud Storage (Best for Remote Teams)

### Google Cloud Storage

```bash
# 1. Create bucket
gsutil mb gs://your-org-scin-data

# 2. Upload data (one time)
gsutil -m cp -r /home/sankar/data/scin gs://your-org-scin-data/
gsutil -m cp -r /home/sankar/models gs://your-org-scin-data/

# 3. Share bucket with team
gsutil iam ch group:team@example.com:objectViewer gs://your-org-scin-data

# 4. Team members download
gsutil -m cp -r gs://your-org-scin-data/scin ~/patient_advocacy_agent/data/
gsutil -m cp -r gs://your-org-scin-data/models ~/patient_advocacy_agent/models/
```

### AWS S3

```bash
# 1. Create bucket
aws s3 mb s3://your-org-scin-data

# 2. Upload data
aws s3 sync /home/sankar/data/scin s3://your-org-scin-data/scin/
aws s3 sync /home/sankar/models s3://your-org-scin-data/models/

# 3. Team members download
aws s3 sync s3://your-org-scin-data/scin ~/patient_advocacy_agent/data/scin/
aws s3 sync s3://your-org-scin-data/models ~/patient_advocacy_agent/models/
```

---

## Method 4: Automated Sync Script (Easy Setup)

Create `scripts/sync_data.sh`:

```bash
#!/bin/bash
# Sync data from central server

set -e

# Configuration
SOURCE_HOST="${SOURCE_HOST:-your-server.example.com}"
SOURCE_USER="${SOURCE_USER:-sankar}"
SOURCE_PATH="${SOURCE_PATH:-/home/sankar}"
LOCAL_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
LOCAL_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"

echo "Syncing SCIN data from ${SOURCE_HOST}..."
rsync -avz --progress "${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_PATH}/data/scin/" "${LOCAL_DATA_DIR}/scin/"

echo "Syncing models from ${SOURCE_HOST}..."
rsync -avz --progress "${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_PATH}/models/" "${LOCAL_MODEL_DIR}/"

echo "✓ Data sync complete!"
echo "Data location: ${LOCAL_DATA_DIR}/scin"
echo "Models location: ${LOCAL_MODEL_DIR}"
```

Usage:
```bash
# Make executable
chmod +x scripts/sync_data.sh

# Run sync (team members)
./scripts/sync_data.sh

# Or with custom source
SOURCE_HOST=your-server.com ./scripts/sync_data.sh
```

---

## Method 5: Docker Container (Best for Reproducibility)

Create `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /workspace/

# Install Python dependencies
RUN pip install -e .

# Create data directories
RUN mkdir -p /data/scin /models

# Copy data and models
COPY data/scin /data/scin
COPY models /models

# Set environment variables
ENV DATA_DIR=/data/scin
ENV MODEL_DIR=/models

CMD ["/bin/bash"]
```

Build and share:
```bash
# Build image
docker build -t your-org/scin-assessment:latest .

# Push to registry
docker push your-org/scin-assessment:latest

# Team members use it
docker pull your-org/scin-assessment:latest
docker run -it your-org/scin-assessment:latest python train_embedder.py
```

---

## Configuration for Different Locations

### Option A: Environment Variables

Add to `.env.local`:
```bash
# Data location (change based on your setup)
DATA_DIR="/mnt/shared_projects/scin_advocacy/scin"
MODEL_DIR="/mnt/shared_projects/scin_advocacy/models"

# Or for cloud
# DATA_DIR="gs://your-org-scin-data/scin"
# MODEL_DIR="gs://your-org-scin-data/models"
```

Update `config.yaml`:
```yaml
data:
  base_dir: ${DATA_DIR:-./data}
  scin_dir: ${DATA_DIR}/scin

models:
  base_dir: ${MODEL_DIR:-./models}
```

### Option B: Update config.yaml for Team

```yaml
# config.yaml - shared team version
data:
  base_dir: /mnt/shared_projects/scin_advocacy/scin
  scin_dir: /mnt/shared_projects/scin_advocacy/scin
  metadata_file: /mnt/shared_projects/scin_advocacy/scin/metadata.csv

models:
  base_dir: /mnt/shared_projects/scin_advocacy/models
  embedder:
    dir: /mnt/shared_projects/scin_advocacy/models/embedder
    final_dir: /mnt/shared_projects/scin_advocacy/models/embedder/final
```

---

## Team Member Quick Start

### For Method 1 (Shared Network Drive)

```bash
# Complete setup for a new team member
git clone <repo-url>
cd patient_advocacy_agent

# Create symlinks
mkdir -p data models
ln -s /mnt/shared_projects/scin_advocacy/scin data/scin
ln -s /mnt/shared_projects/scin_advocacy/models/* models/

# Verify
python -c "from patient_advocacy_agent import SCINDataLoader; print('✓ Setup successful!')"
```

### For Method 2 (Git LFS)

```bash
git clone <repo-url>
cd patient_advocacy_agent
git lfs pull
python -c "from patient_advocacy_agent import SCINDataLoader; print('✓ Setup successful!')"
```

### For Method 4 (Sync Script)

```bash
git clone <repo-url>
cd patient_advocacy_agent
./scripts/sync_data.sh
python -c "from patient_advocacy_agent import SCINDataLoader; print('✓ Setup successful!')"
```

---

## Comparison Table

| Method | Cost | Setup Time | Speed | Flexibility | Remote Access |
|--------|------|-----------|-------|-------------|--------------|
| **Shared Network** | Low | 30 min | Very Fast | High | Limited |
| **Git LFS** | Medium | 20 min | Medium | High | Good |
| **Google Cloud** | Medium | 15 min | Medium | High | Excellent |
| **AWS S3** | Medium | 15 min | Medium | High | Excellent |
| **Sync Script** | Low | 10 min | Fast | Very High | Good |
| **Docker** | Medium | 45 min | Medium | Very High | Excellent |

---

## Recommendations by Scenario

### Small Local Team (< 5 people)
→ **Shared Network Drive** (Method 1)
- Easiest to set up
- No cloud costs
- Fast file access

### Team with Version Control
→ **Git LFS** (Method 2)
- Track data versions
- Integrate with code
- CI/CD friendly

### Remote or Distributed Team
→ **Google Cloud Storage** (Method 3)
- Accessible anywhere
- Automatic backups
- Easy permission management

### Need Reproducibility
→ **Docker** (Method 5)
- Same environment everywhere
- Easy deployment
- No installation needed

### Maximum Flexibility
→ **Sync Script** (Method 4)
- Works with any storage
- Minimal dependencies
- Easy to customize

---

## Troubleshooting

### Problem: "Permission denied" when accessing shared data

```bash
# Check permissions
ls -la /mnt/shared_projects/scin_advocacy/

# Fix: Ask admin to run
sudo chmod -R g+rx /mnt/shared_projects/scin_advocacy/
```

### Problem: Git LFS quota exceeded

```bash
# Check usage
git lfs ls-files | wc -l

# Prune old versions
git lfs prune

# Increase quota on server
```

### Problem: Slow data download

```bash
# Check connection
time head -c 100M /mnt/shared_projects/scin_advocacy/scin/metadata.csv > /dev/null

# Use parallel download
gsutil -m cp -r gs://your-bucket/data/* local_path/  # For GCS
aws s3 sync s3://bucket/data local_path/ --parallel 10  # For S3
```

---

## Next Steps

1. **Choose a method** based on your team's needs
2. **Set up storage** using the guide for your chosen method
3. **Share credentials/paths** with team members
4. **Test** with one team member first
5. **Document** any customizations in your team wiki

For questions or issues, contact your data administrator.
