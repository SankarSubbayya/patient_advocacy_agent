# SCIN Dataset Download Options

This guide explains the three ways to download the SCIN (Skin Condition Image Network) dataset for the Patient Advocacy Agent.

---

## The SCIN Dataset Issue

The original SCIN repository URL referenced in documentation (`https://github.com/ISMAE-SUDA/SCIN`) **no longer exists**.

The **official SCIN dataset** is now hosted by **Google Research** at:
- **Repository**: https://github.com/google-research-datasets/scin
- **Storage**: Google Cloud Storage bucket `dx-scin-public-data`
- **Access**: Open access for research use
- **Size**: 10,000+ dermatology images
- **Paper**: https://arxiv.org/abs/2402.18545

---

## Option 1: Download from Google Cloud Storage (Recommended)

This is the **official and most direct method** to get the real SCIN dataset.

### Prerequisites

```bash
pip install google-cloud-storage pandas pillow
```

Or with uv:
```bash
uv pip install google-cloud-storage pandas pillow
```

### Run Download

```bash
python download_scin_gcs.py
```

### What It Does

1. Connects to Google Cloud Storage bucket: `dx-scin-public-data`
2. Downloads official metadata CSV (`dataset/scin_cases.csv`)
3. Downloads all images from GCS
4. Organizes images by condition in `data/scin/images/`
5. Creates `metadata.csv` for compatibility with your training pipeline

### Timeline

- **Download**: 30-60 minutes (10,000+ images, ~500 MB)
- **Organization**: 5-10 minutes
- **Total**: 35-70 minutes

### Pros & Cons

✓ **Pros:**
- Official SCIN dataset from Google Research
- Complete with 10,000+ real images
- Includes proper metadata
- Follows proper attribution

✗ **Cons:**
- Requires internet connection
- Slower than alternatives (many small file downloads)
- Requires `google-cloud-storage` library
- Network reliability dependent

---

## Option 2: Use Alternative Dataset (HAM10000)

If you want **faster setup** or have **issues with GCS**, use **HAM10000** from Kaggle.

### Benefits

- **Size**: 10,015 images (similar to SCIN)
- **Quality**: Well-organized, widely used
- **Speed**: Direct download from Kaggle
- **Format**: Already organized with metadata
- **No dependencies**: Just `wget` or browser download

### Steps

#### Step 1: Download from Kaggle

**Option A: Using Kaggle API (Recommended)**

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Extract
unzip skin-cancer-mnist-ham10000.zip -d ./data/ham10000/
```

**Option B: Manual Download**

1. Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Click "Download"
3. Extract to `data/ham10000/`

#### Step 2: Prepare for Training

```bash
# Copy to expected location
cp -r data/ham10000 data/scin
```

### Structure

```
data/scin/images/
├── AKIEC/    (Actinic Keratosis)
├── BCC/      (Basal Cell Carcinoma)
├── MEL/      (Melanoma)
├── NV/       (Nevus)
├── SCC/      (Squamous Cell Carcinoma)
└── ...
```

### Training Compatibility

Update `train_embedder.py` to handle the HAM10000 structure:

```python
# In train_embedder.py, modify the condition mapping:
condition_mapping = {
    'AKIEC': 'Actinic Keratosis',
    'BCC': 'Basal Cell Carcinoma',
    'MEL': 'Melanoma',
    'NV': 'Nevus',
    'SCC': 'Squamous Cell Carcinoma',
    'VASC': 'Vascular',
    'DF': 'Dermatofibroma'
}
```

---

## Option 3: Use Sample Dataset (Development Only)

For **quick testing and development**, use the included sample dataset.

### Run

```bash
# The sample dataset is automatically created if you run:
python verify_setup.py
```

### Contents

- **80 sample images** (10 per condition)
- **8 conditions**: Acne, Dermatitis, Eczema, Melanoma, Psoriasis, Rosacea, Urticaria, Vitiligo
- **Training time**: 10-30 minutes (vs 2-8 hours for full dataset)

### Use Case

This option is useful for:
- Testing the pipeline
- Debugging code
- Quick validation
- Development and CI/CD

### Limitations

- Too small for real-world accuracy
- Not suitable for production
- Only for testing/development

---

## Comparison Table

| Feature | GCS (SCIN) | HAM10000 | Sample |
|---------|-----------|----------|--------|
| **Images** | 10,000+ | 10,015 | 80 |
| **Quality** | Official | Well-used | Synthetic |
| **Speed** | 35-70 min | 10-30 min | <1 min |
| **Dependencies** | google-cloud-storage | kaggle (optional) | None |
| **For Production** | ✓ Yes | ✓ Yes | ✗ No |
| **Official Dataset** | ✓ Yes | ✗ No | ✗ No |
| **Setup Difficulty** | Medium | Easy | Easy |

---

## Recommended Workflow

### For Production/Research

```bash
# 1. Install dependencies
pip install google-cloud-storage pandas pillow

# 2. Download official SCIN from GCS
python download_scin_gcs.py

# 3. Train embedder (2-8 hours)
uv run python train_embedder.py

# 4. Build index (15-30 minutes)
uv run python build_index.py

# 5. Deploy
uv run python example_usage.py
```

**Total Time**: ~3-10 hours

### For Development/Testing

```bash
# 1. Use sample dataset (automatic)
uv run python verify_setup.py

# 2. Quick training test (30 minutes)
# Edit train_embedder.py to use fewer epochs:
#   num_epochs = 2

uv run python train_embedder.py

# 3. Fast index build (5 minutes)
uv run python build_index.py

# 4. Test
uv run python example_usage.py
```

**Total Time**: ~1 hour

---

## Troubleshooting

### GCS Download Issues

**Problem: "HTTP 404: Not Found"**
- The old ISMAE-SUDA URL no longer works
- Use this script instead: `download_scin_gcs.py`

**Problem: "google-cloud-storage not found"**
```bash
pip install google-cloud-storage
# Or with uv
uv pip install google-cloud-storage
```

**Problem: Slow download speed**
- This is normal for 10,000+ small files
- GCS streaming downloads are inherently slower
- Consider using HAM10000 instead for faster setup

### GCS Authentication Issues

**Public bucket access** (dx-scin-public-data) should work without authentication. If you get auth errors:

1. Ensure `google-cloud-storage` is installed
2. Check internet connection
3. Try using HAM10000 instead

### Kaggle API Issues

**Problem: "Kaggle API key not found"**
```bash
# Create Kaggle API key:
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Save to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

---

## File Locations

After downloading, images will be organized as:

```
data/scin/
├── images/
│   ├── Acne/
│   │   ├── image_001.jpg
│   │   └── ...
│   ├── Eczema/
│   ├── Psoriasis/
│   └── ...
└── metadata.csv
```

All three download methods create this same structure, so the rest of your pipeline works identically.

---

## Next Steps

Once dataset is downloaded, follow the [Data Pipeline Guide](docs/DATA_PIPELINE_GUIDE.md):

1. **Train Embedder**: `uv run python train_embedder.py`
2. **Build Index**: `uv run python build_index.py`
3. **Run Assessments**: `uv run python example_usage.py`

---

## References

- **Official SCIN**: https://github.com/google-research-datasets/scin
- **Paper**: https://arxiv.org/abs/2402.18545
- **HAM10000**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Google Cloud Storage**: https://cloud.google.com/python/docs/reference/storage

---

**Last Updated**: November 2024
**Status**: Complete dataset download guide
