# Documentation Index

Complete guide to all documentation files in the Patient Advocacy Agent project.

---

## Quick Navigation

### 🚀 Getting Started (Start Here!)

| Document | Description | Read Time |
|----------|-------------|-----------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | **Complete setup guide** - Installation, verification, data pipeline, training, and usage. Start here! | 15 min |
| [QUICKSTART.md](QUICKSTART.md) | **5-minute quick start** - Minimal example to get running in minutes | 5 min |

### 📖 Core Documentation

| Document | Description | Read Time |
|----------|-------------|-----------|
| [README.md](README.md) | **Project overview** - Features, architecture, components, and API | 10 min |
| [ARCHITECTURE.md](ARCHITECTURE.md) | **System design** - How components work together, data flow, technical decisions | 20 min |

### 🛠️ Setup & Configuration

| Document | Description | Read Time |
|----------|-------------|-----------|
| [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) | **Environment configuration** - Python setup, uv package manager, dependency management | 15 min |
| [SETUP_COMPLETE.md](SETUP_COMPLETE.md) | **Setup verification checklist** - Verify your installation is correct | 5 min |

### 🎯 Implementation Guides

| Document | Description | Read Time |
|----------|-------------|-----------|
| [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md) | **Complete data pipeline** - Download SCIN, prepare data, create metadata, validate dataset | 20 min |
| [SCRIPTS_SUMMARY.md](SCRIPTS_SUMMARY.md) | **Helper scripts reference** - Overview of all 5 scripts with usage examples | 10 min |

### 🧠 Model & Training

| Document | Description | Read Time |
|----------|-------------|-----------|
| [SIGLIP_MODELS.md](SIGLIP_MODELS.md) | **SigLIP model selection** - Base vs Large comparison, specs, when to use each | 15 min |
| [MODEL_STORAGE_GUIDE.md](MODEL_STORAGE_GUIDE.md) | **Model storage & loading** - Where models are saved, directory structure, loading code | 10 min |
| [MPS_APPLE_SILICON.md](MPS_APPLE_SILICON.md) | **GPU acceleration** - Use Apple Silicon MPS for 2-3x faster training | 15 min |

---

## Documentation by Purpose

### 👤 I'm a New User

**Read in this order:**
1. Start: [GETTING_STARTED.md](GETTING_STARTED.md) ← Complete walkthrough
2. Quick test: [QUICKSTART.md](QUICKSTART.md) ← 5-minute example
3. Understand: [README.md](README.md) ← What it does

**Then proceed based on your goal:**
- **To train the model**: [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md)
- **To understand the system**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **To use API**: See API section in [README.md](README.md)

---

### 🔧 I Want to Set Up the Environment

1. [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) - Detailed setup guide
2. [GETTING_STARTED.md](GETTING_STARTED.md) - Installation section
3. [SETUP_COMPLETE.md](SETUP_COMPLETE.md) - Verify it worked

---

### 📊 I Have Data and Want to Train

1. [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md) - Download & prepare data
2. [GETTING_STARTED.md](GETTING_STARTED.md) - Training section
3. [SIGLIP_MODELS.md](SIGLIP_MODELS.md) - Model selection
4. [MPS_APPLE_SILICON.md](MPS_APPLE_SILICON.md) - GPU acceleration (optional)

---

### ⚡ I Have Apple Silicon (M1/M2/M3) and Want Fast Training

1. [MPS_APPLE_SILICON.md](MPS_APPLE_SILICON.md) - Complete GPU guide
2. [GETTING_STARTED.md](GETTING_STARTED.md) - Training section
3. [SIGLIP_MODELS.md](SIGLIP_MODELS.md) - Model specs

Expected speedup: **2-3x faster** than CPU!

---

### 🏗️ I Want to Understand the Architecture

1. [README.md](README.md) - Overview of components
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Deep dive into design
3. [MODEL_STORAGE_GUIDE.md](MODEL_STORAGE_GUIDE.md) - How models are organized

---

### 🚀 I Want to Deploy This

1. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the system
2. [README.md](README.md) - See REST API section
3. [GETTING_STARTED.md](GETTING_STARTED.md) - Learn deployment section
4. [MODEL_STORAGE_GUIDE.md](MODEL_STORAGE_GUIDE.md) - Model paths for deployment

---

### 🐛 Something's Not Working

Check [GETTING_STARTED.md](GETTING_STARTED.md#troubleshooting) for solutions, or:

| Issue | Document |
|-------|----------|
| Import errors | [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) |
| Training crashes | [GETTING_STARTED.md](GETTING_STARTED.md#troubleshooting) |
| Dataset problems | [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md#troubleshooting) |
| GPU/MPS issues | [MPS_APPLE_SILICON.md](MPS_APPLE_SILICON.md#troubleshooting-mps-issues) |
| Model loading fails | [MODEL_STORAGE_GUIDE.md](MODEL_STORAGE_GUIDE.md) |

---

## Documentation Overview

### What Each File Contains

#### [README.md](README.md)
- Project overview and features
- Installation instructions
- Architecture diagram
- Component descriptions
- API usage examples
- REST API endpoints
- Performance notes

#### [GETTING_STARTED.md](GETTING_STARTED.md) ⭐
- Complete setup and installation
- Step-by-step data pipeline
- Training instructions
- Index building
- Usage examples
- Common commands
- Troubleshooting guide
- Performance benchmarks
- **START HERE!**

#### [QUICKSTART.md](QUICKSTART.md)
- 5-minute quick start
- Minimal working example
- Demo code
- Basic usage

#### [ARCHITECTURE.md](ARCHITECTURE.md)
- System design and components
- Data flow diagrams
- Technical decisions
- Component interactions
- API structure
- Module descriptions
- Design patterns

#### [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
- Python environment setup
- Virtual environment creation
- Dependency management with uv
- Package installation
- Troubleshooting setup issues

#### [SETUP_COMPLETE.md](SETUP_COMPLETE.md)
- Setup verification checklist
- Installation confirmation
- Component testing
- Next steps

#### [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md)
- SCIN dataset overview and download
- Data preparation and organization
- Metadata creation
- Dataset validation
- Alternative datasets (ISIC, DermNet, etc.)
- Complete workflow timeline
- Troubleshooting data issues

#### [SCRIPTS_SUMMARY.md](SCRIPTS_SUMMARY.md)
- Overview of 5 helper scripts
- Purpose and usage of each
- Input/output descriptions
- Configuration options
- Timeline estimates
- Troubleshooting script issues

#### [SIGLIP_MODELS.md](SIGLIP_MODELS.md)
- SigLIP model variants comparison
- Base vs Large specifications
- Performance benchmarks
- Memory requirements
- When to use each model
- Upgrading between models

#### [MODEL_STORAGE_GUIDE.md](MODEL_STORAGE_GUIDE.md)
- Model storage locations
- Directory structure
- Saving trained models
- Loading models in code
- Model versioning
- Backup and recovery
- Storage optimization

#### [MPS_APPLE_SILICON.md](MPS_APPLE_SILICON.md)
- Apple Silicon MPS GPU support
- Availability checking
- Configuration and optimization
- Performance benchmarks (2-3x faster!)
- Training examples with MPS
- Troubleshooting MPS issues
- FAQ about MPS

#### [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- High-level implementation overview
- What was implemented
- Technical approaches
- Key decisions

#### [STATUS.md](STATUS.md)
- Current project status
- Feature completion status
- Known issues
- Roadmap

---

## Document Statistics

| Document | Size | Lines | Type |
|----------|------|-------|------|
| GETTING_STARTED.md | 13.2 KB | 600+ | Setup Guide |
| ARCHITECTURE.md | 22.4 KB | 850+ | Design |
| DATA_PIPELINE_GUIDE.md | 13.1 KB | 540+ | Guide |
| MPS_APPLE_SILICON.md | 12.3 KB | 579+ | Guide |
| MODEL_STORAGE_GUIDE.md | 18.0 KB | 680+ | Guide |
| SIGLIP_MODELS.md | 14.0 KB | 550+ | Reference |
| README.md | 10.6 KB | 420+ | Overview |
| QUICKSTART.md | 8.8 KB | 350+ | Tutorial |
| SCRIPTS_SUMMARY.md | 8.5 KB | 420+ | Reference |
| ENVIRONMENT_SETUP.md | 11.1 KB | 440+ | Setup |
| SETUP_COMPLETE.md | 7.6 KB | 390+ | Checklist |

**Total Documentation**: ~130 KB, 5000+ lines of comprehensive guides

---

## Reading Guide by Experience Level

### Beginner (No ML experience)

**Essential reads:**
1. [GETTING_STARTED.md](GETTING_STARTED.md) - Understand the basics
2. [QUICKSTART.md](QUICKSTART.md) - See it work
3. [README.md](README.md) - Learn what each component does

**Optional:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - Understand how it fits together

### Intermediate (Some ML experience)

**Essential reads:**
1. [GETTING_STARTED.md](GETTING_STARTED.md) - Setup and workflow
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the design
3. [SIGLIP_MODELS.md](SIGLIP_MODELS.md) - Model selection

**Depending on your goal:**
- Training: [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md) + [MPS_APPLE_SILICON.md](MPS_APPLE_SILICON.md)
- Deployment: [README.md](README.md) API section
- Storage: [MODEL_STORAGE_GUIDE.md](MODEL_STORAGE_GUIDE.md)

### Advanced (ML/ML Engineer)

**Quick reference:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions
- [SCRIPTS_SUMMARY.md](SCRIPTS_SUMMARY.md) - Script details
- [MODEL_STORAGE_GUIDE.md](MODEL_STORAGE_GUIDE.md) - Model management

**Customization:**
- [SIGLIP_MODELS.md](SIGLIP_MODELS.md) - Model variants
- [MPS_APPLE_SILICON.md](MPS_APPLE_SILICON.md) - Optimization
- [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md) - Data alternatives

---

## Common Questions & Where to Find Answers

| Question | Document |
|----------|----------|
| How do I get started? | [GETTING_STARTED.md](GETTING_STARTED.md) |
| What does this system do? | [README.md](README.md) |
| How does it work? | [ARCHITECTURE.md](ARCHITECTURE.md) |
| How do I set up the environment? | [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) |
| How do I download the data? | [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md) |
| How do I train the model? | [GETTING_STARTED.md](GETTING_STARTED.md#training) |
| How do I use MPS/GPU? | [MPS_APPLE_SILICON.md](MPS_APPLE_SILICON.md) |
| Which SigLIP model should I use? | [SIGLIP_MODELS.md](SIGLIP_MODELS.md) |
| Where are models stored? | [MODEL_STORAGE_GUIDE.md](MODEL_STORAGE_GUIDE.md) |
| How do I use the API? | [README.md](README.md#api-usage) |
| What do the scripts do? | [SCRIPTS_SUMMARY.md](SCRIPTS_SUMMARY.md) |
| Something's broken | [GETTING_STARTED.md#troubleshooting](GETTING_STARTED.md#troubleshooting) |

---

## Related Files (Not Documentation)

### Source Code
- `src/patient_advocacy_agent/` - All Python modules
- `src/patient_advocacy_agent/__init__.py` - Package exports
- Individual modules: `data.py`, `embedder.py`, `clustering.py`, `rag.py`, `agent.py`, `api.py`

### Scripts
- `download_scin_dataset.py` - Dataset downloader
- `train_embedder.py` - Model trainer
- `build_index.py` - Index builder
- `verify_setup.py` - Environment verifier
- `example_usage.py` - Demo workflow

### Configuration
- `pyproject.toml` - Project configuration
- `.env` - Environment variables
- `.python-version` - Python version specification
- `config.yaml` - Application configuration

---

## Best Practices for Reading Documentation

1. **Start with GETTING_STARTED.md** - It provides the complete context
2. **Follow the workflow order** - Each guide builds on previous ones
3. **Use the troubleshooting sections** - They're at the end of each guide
4. **Check related documents** - Links between documents guide you
5. **Use the index** - This document to find what you need

---

## Version Information

- **Documentation Version**: 1.0
- **Package Version**: 0.1.0
- **Python Version**: 3.12+
- **Last Updated**: 2024
- **Status**: Complete and Production Ready

---

## Next Steps

### New User?
→ Read [GETTING_STARTED.md](GETTING_STARTED.md) and start with the installation section.

### Need Specific Help?
→ Use the "Common Questions" table above to find your answer.

### Want to Contribute?
→ Check [STATUS.md](STATUS.md) for what's needed next.

### Have Questions?
→ Check the troubleshooting section in the relevant document.

---

**Happy learning!** 🚀

Start with [GETTING_STARTED.md](GETTING_STARTED.md) →
