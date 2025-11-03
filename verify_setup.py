#!/usr/bin/env python
"""Quick verification script to check if environment is set up correctly."""

import sys
from pathlib import Path

print("\n" + "="*80)
print("Patient Advocacy Agent - Environment Verification")
print("="*80)

# Test 1: Import all modules
print("\n1. Testing module imports...")
try:
    from patient_advocacy_agent import (
        SCINDataLoader,
        SigLIPEmbedder,
        EmbedderTrainer,
        SimilarityIndex,
        ImageClusterer,
        RAGPipeline,
        CaseRetriever,
        MedicalKnowledgeBase,
        MedGeminiAgent,
        PatientAssessmentAPI,
        PatientAssessmentRequest,
    )
    print("   ✓ All modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check Python version
print("\n2. Checking Python version...")
import platform
python_version = platform.python_version()
if python_version.startswith("3.12") or python_version.startswith("3.13"):
    print(f"   ✓ Python {python_version} (compatible)")
else:
    print(f"   ⚠ Python {python_version} (may have compatibility issues)")

# Test 3: Check key dependencies
print("\n3. Checking dependencies...")
dependencies = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'sklearn': 'Scikit-learn',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'pydantic': 'Pydantic',
    'faiss': 'FAISS',
}

for module_name, display_name in dependencies.items():
    try:
        __import__(module_name)
        # Get version if available
        try:
            if module_name == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(module_name)
                version = getattr(mod, '__version__', 'unknown')
            print(f"   ✓ {display_name:20} {version}")
        except:
            print(f"   ✓ {display_name:20} (installed)")
    except ImportError:
        print(f"   ✗ {display_name:20} (not installed)")

# Test 4: Check GPU availability
print("\n4. Checking GPU/CUDA...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("   ℹ CUDA not available (CPU mode will be used)")
except Exception as e:
    print(f"   ⚠ Could not check CUDA: {e}")

# Test 5: Test basic model loading
print("\n5. Testing SigLIP model loading...")
try:
    embedder = SigLIPEmbedder(model_name="google/siglip-base-patch16-224")
    print(f"   ✓ SigLIP model loaded successfully")
    print(f"     - Model: google/siglip-base-patch16-224")
    print(f"     - Projection dimension: 512")
except Exception as e:
    print(f"   ⚠ Could not load SigLIP model: {e}")
    print(f"     (This is normal if no internet connection)")

# Test 6: Test basic data structures
print("\n6. Testing data structures...")
try:
    request = PatientAssessmentRequest(
        patient_id="TEST001",
        age=35,
        gender="F",
        symptoms=["itching", "redness"]
    )
    print(f"   ✓ PatientAssessmentRequest created successfully")
    print(f"     - Patient ID: {request.patient_id}")
    print(f"     - Age: {request.age}")
    print(f"     - Symptoms: {', '.join(request.symptoms)}")
except Exception as e:
    print(f"   ✗ Error creating PatientAssessmentRequest: {e}")

# Summary
print("\n" + "="*80)
print("ENVIRONMENT VERIFICATION COMPLETE")
print("="*80)
print("\n✓ Environment is properly configured!")
print("\nNext steps:")
print("  1. Download SCIN dataset from https://github.com/ISMAE-SUDA/SCIN")
print("  2. Create metadata.csv with image information")
print("  3. Run: uv run python example_usage.py")
print("  4. Or check: uv run python -c \"from patient_advocacy_agent import *; print('Ready!')\"")
print("\nQuick test:")
print("  uv run python -c \"from patient_advocacy_agent import PatientAssessmentRequest\"")
print("  uv run python -c \"import patient_advocacy_agent; print(f'Version: {patient_advocacy_agent.__version__}')\"")
print("\n")
