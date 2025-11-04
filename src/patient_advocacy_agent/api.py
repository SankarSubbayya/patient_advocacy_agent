"""API interface for patient assessment and physician report generation."""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import torch
from pydantic import BaseModel, Field, validator
from PIL import Image
import io

# Agent-related classes have been removed from this project
# These imports are no longer available:
# from .agent import (
#     MedGeminiAgent,
#     PatientCase,
#     AssessmentResult,
#     PhysicianReport
# )

logger = logging.getLogger(__name__)


class PatientAssessmentRequest(BaseModel):
    """Request model for patient assessment."""
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=150, description="Patient age")
    gender: str = Field(..., description="Patient gender (M/F/Other)")
    symptoms: List[str] = Field(..., min_items=1, description="List of symptoms")
    symptom_onset: Optional[str] = Field(None, description="When symptoms started")
    patient_notes: Optional[str] = Field(None, description="Additional patient notes")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")

    @validator('gender')
    def validate_gender(cls, v):
        if v.lower() not in ['m', 'f', 'other', 'male', 'female']:
            raise ValueError('Invalid gender')
        return v


class ReportExportRequest(BaseModel):
    """Request model for exporting report."""
    report_id: str
    format: str = Field("json", description="Export format (json, pdf, txt)")
    include_similar_cases: bool = Field(True, description="Include similar cases in report")
    include_knowledge_summary: bool = Field(True, description="Include knowledge summary")


class PatientAssessmentAPI:
    """API interface for patient assessment system.

    Note: The agent layer has been removed from this system.
    This API now only handles embeddings and retrieval.
    """

    def __init__(
        self,
        embedder,
        storage_dir: Optional[Path] = None
    ):
        """
        Initialize API.

        Args:
            embedder: SigLIPEmbedder instance
            storage_dir: Directory to store reports and assessments
        """
        self.embedder = embedder
        self.storage_dir = Path(storage_dir) if storage_dir else Path.home() / ".patient_advocacy_reports"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Cache for recent assessments
        self.assessment_cache = {}

    def assess_patient(
        self,
        request: PatientAssessmentRequest,
        confidence_threshold: float = 0.3,
        num_similar_cases: int = 5
    ) -> Dict[str, Any]:
        """
        Assess a patient and generate recommendations.

        NOTE: The agent layer has been removed. This method is deprecated.
        For embeddings, use extract_embeddings() instead.

        Args:
            request: Patient assessment request
            confidence_threshold: Minimum confidence threshold
            num_similar_cases: Number of similar cases to retrieve

        Returns:
            Error response (agent functionality removed)
        """
        logger.warning("assess_patient() called but agent layer has been removed")
        return {
            'status': 'error',
            'patient_id': request.patient_id,
            'error': 'Agent layer has been removed from this system. Use embeddings directly.',
            'timestamp': datetime.now().isoformat()
        }

    def generate_physician_report(
        self,
        patient_id: str,
        assessment: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate physician report for a patient.

        NOTE: The agent layer has been removed. This method is deprecated.

        Args:
            patient_id: Patient ID
            assessment: Optional cached assessment result

        Returns:
            Error response (agent functionality removed)
        """
        logger.warning("generate_physician_report() called but agent layer has been removed")
        return {
            'status': 'error',
            'patient_id': patient_id,
            'error': 'Agent layer has been removed from this system. Use embeddings directly.',
            'timestamp': datetime.now().isoformat()
        }

    def export_report(
        self,
        request: ReportExportRequest
    ) -> Dict[str, Any]:
        """
        Export report in specified format.

        NOTE: The agent layer has been removed. This method is deprecated.

        Args:
            request: Report export request

        Returns:
            Error response (agent functionality removed)
        """
        logger.warning("export_report() called but agent layer has been removed")
        return {
            'status': 'error',
            'error': 'Agent layer has been removed from this system.'
        }

    def get_assessment_history(self, patient_id: str) -> Dict[str, Any]:
        """
        Get assessment history for a patient.

        NOTE: The agent layer has been removed. This method is deprecated.

        Args:
            patient_id: Patient ID

        Returns:
            Error response (agent functionality removed)
        """
        logger.warning("get_assessment_history() called but agent layer has been removed")
        return {
            'status': 'error',
            'error': 'Agent layer has been removed from this system.'
        }

    def _process_image_data(self, image_data: str) -> torch.Tensor:
        """
        Process base64 encoded image data to tensor.

        Args:
            image_data: Base64 encoded image data

        Returns:
            Image tensor
        """
        import base64
        from torchvision import transforms

        # Decode base64
        image_bytes = base64.b64decode(image_data)

        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Transform to tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transform(image)
