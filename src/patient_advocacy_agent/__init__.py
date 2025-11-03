#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#   This code is part of the training material and, therefore, part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
from svlearn.config.configuration import ConfigurationMixin

from dotenv import load_dotenv
load_dotenv()

config = ConfigurationMixin().load_config()

# Import core modules
from .data import (
    SkinConditionDataset,
    SCINDataLoader,
    ImageMetadata
)

from .embedder import (
    SigLIPEmbedder,
    EmbedderTrainer,
    ContrastiveLoss
)

from .clustering import (
    SimilarityIndex,
    ImageClusterer,
    ConditionBasedGrouping,
    ClusterResult
)

from .rag import (
    RAGPipeline,
    CaseRetriever,
    MedicalKnowledgeBase,
    RetrievedCase
)

from .agent import (
    MedGeminiAgent,
    PatientCase,
    AssessmentResult,
    PhysicianReport
)

from .api import (
    PatientAssessmentAPI,
    PatientAssessmentRequest,
    ReportExportRequest
)

__version__ = "0.1.0"
__all__ = [
    "SkinConditionDataset",
    "SCINDataLoader",
    "ImageMetadata",
    "SigLIPEmbedder",
    "EmbedderTrainer",
    "ContrastiveLoss",
    "SimilarityIndex",
    "ImageClusterer",
    "ConditionBasedGrouping",
    "ClusterResult",
    "RAGPipeline",
    "CaseRetriever",
    "MedicalKnowledgeBase",
    "RetrievedCase",
    "MedGeminiAgent",
    "PatientCase",
    "AssessmentResult",
    "PhysicianReport",
    "PatientAssessmentAPI",
    "PatientAssessmentRequest",
    "ReportExportRequest"
]
