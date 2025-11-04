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

    def to_patient_case(self) -> PatientCase:
        """Convert request to PatientCase."""
        return PatientCase(
            patient_id=self.patient_id,
            age=self.age,
            gender=self.gender,
            symptoms=self.symptoms,
            symptom_onset=self.symptom_onset,
            patient_notes=self.patient_notes
        )


class ReportExportRequest(BaseModel):
    """Request model for exporting report."""
    report_id: str
    format: str = Field("json", description="Export format (json, pdf, txt)")
    include_similar_cases: bool = Field(True, description="Include similar cases in report")
    include_knowledge_summary: bool = Field(True, description="Include knowledge summary")


class PatientAssessmentAPI:
    """API interface for patient assessment system."""

    def __init__(
        self,
        agent,  # MedGeminiAgent (now removed from project)
        embedder,
        storage_dir: Optional[Path] = None
    ):
        """
        Initialize API.

        Args:
            agent: Agent instance (MedGeminiAgent has been removed)
            embedder: SigLIPEmbedder instance
            storage_dir: Directory to store reports and assessments
        """
        self.agent = agent
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

        Args:
            request: Patient assessment request
            confidence_threshold: Minimum confidence threshold
            num_similar_cases: Number of similar cases to retrieve

        Returns:
            Assessment result as dictionary
        """
        logger.info(f"Processing assessment request for patient {request.patient_id}")

        try:
            # Convert request to patient case
            patient_case = request.to_patient_case()

            # Process image if provided
            image_tensor = None
            if request.image_data:
                image_tensor = self._process_image_data(request.image_data)

            # Run assessment
            assessment = self.agent.assess_patient(
                patient_case=patient_case,
                image_tensor=image_tensor,
                num_similar_cases=num_similar_cases,
                confidence_threshold=confidence_threshold
            )

            # Cache assessment
            self.assessment_cache[request.patient_id] = assessment

            # Save assessment to disk
            self._save_assessment(assessment)

            return {
                'status': 'success',
                'patient_id': request.patient_id,
                'assessment': assessment.model_dump(),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Assessment failed for patient {request.patient_id}: {str(e)}")
            return {
                'status': 'error',
                'patient_id': request.patient_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def generate_physician_report(
        self,
        patient_id: str,
        assessment: Optional[AssessmentResult] = None
    ) -> Dict[str, Any]:
        """
        Generate physician report for a patient.

        Args:
            patient_id: Patient ID
            assessment: Optional cached assessment result

        Returns:
            Generated physician report as dictionary
        """
        logger.info(f"Generating physician report for patient {patient_id}")

        try:
            # Get assessment from cache or storage
            if assessment is None:
                assessment = self._load_assessment(patient_id)

            if assessment is None:
                return {
                    'status': 'error',
                    'error': f'No assessment found for patient {patient_id}',
                    'timestamp': datetime.now().isoformat()
                }

            # Create patient case from assessment metadata
            patient_case = PatientCase(
                patient_id=patient_id,
                age=0,  # Would need to be stored with assessment
                gender='Unknown',  # Would need to be stored with assessment
                symptoms=assessment.assessment_date.split(',') if assessment.assessment_date else []
            )

            # Generate report
            report = self.agent.generate_physician_report(assessment, patient_case)

            # Save report
            self._save_report(report)

            return {
                'status': 'success',
                'report': report.model_dump(),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Report generation failed for patient {patient_id}: {str(e)}")
            return {
                'status': 'error',
                'patient_id': patient_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def export_report(
        self,
        request: ReportExportRequest
    ) -> Dict[str, Any]:
        """
        Export report in specified format.

        Args:
            request: Report export request

        Returns:
            Exported report data
        """
        logger.info(f"Exporting report {request.report_id} as {request.format}")

        try:
            # Load report
            report = self._load_report(request.report_id)

            if report is None:
                return {
                    'status': 'error',
                    'error': f'Report {request.report_id} not found'
                }

            # Export based on format
            if request.format.lower() == 'json':
                return {
                    'status': 'success',
                    'format': 'json',
                    'content': report.model_dump(),
                    'data_type': 'object'
                }

            elif request.format.lower() == 'txt':
                content = self._format_report_as_text(report)
                return {
                    'status': 'success',
                    'format': 'txt',
                    'content': content,
                    'data_type': 'text'
                }

            elif request.format.lower() == 'pdf':
                # PDF generation would require additional library (e.g., reportlab)
                # For now, return text format
                content = self._format_report_as_text(report)
                return {
                    'status': 'success',
                    'format': 'pdf',
                    'content': content,
                    'note': 'PDF generation requires additional dependencies'
                }

            else:
                return {
                    'status': 'error',
                    'error': f'Unsupported format: {request.format}'
                }

        except Exception as e:
            logger.error(f"Report export failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def get_assessment_history(self, patient_id: str) -> Dict[str, Any]:
        """
        Get assessment history for a patient.

        Args:
            patient_id: Patient ID

        Returns:
            List of assessments for patient
        """
        try:
            assessments = []
            patient_dir = self.storage_dir / "assessments" / patient_id

            if patient_dir.exists():
                for assessment_file in patient_dir.glob("*.json"):
                    with open(assessment_file) as f:
                        assessment_data = json.load(f)
                        assessments.append(assessment_data)

            return {
                'status': 'success',
                'patient_id': patient_id,
                'assessments': assessments,
                'count': len(assessments)
            }

        except Exception as e:
            logger.error(f"Failed to retrieve assessment history: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
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

    def _save_assessment(self, assessment: AssessmentResult) -> None:
        """Save assessment to disk."""
        patient_dir = self.storage_dir / "assessments" / assessment.patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        filename = f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = patient_dir / filename

        with open(filepath, 'w') as f:
            json.dump(assessment.model_dump(), f, indent=2)

        logger.info(f"Assessment saved to {filepath}")

    def _save_report(self, report: PhysicianReport) -> None:
        """Save report to disk."""
        reports_dir = self.storage_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        filepath = reports_dir / f"{report.report_id}.json"

        with open(filepath, 'w') as f:
            json.dump(report.model_dump(), f, indent=2)

        logger.info(f"Report saved to {filepath}")

    def _load_assessment(self, patient_id: str) -> Optional[AssessmentResult]:
        """Load most recent assessment for patient."""
        patient_dir = self.storage_dir / "assessments" / patient_id

        if not patient_dir.exists():
            return None

        # Get most recent assessment
        assessment_files = sorted(patient_dir.glob("*.json"), reverse=True)

        if not assessment_files:
            return None

        with open(assessment_files[0]) as f:
            data = json.load(f)
            return AssessmentResult(**data)

    def _load_report(self, report_id: str) -> Optional[PhysicianReport]:
        """Load report by ID."""
        filepath = self.storage_dir / "reports" / f"{report_id}.json"

        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = json.load(f)
            return PhysicianReport(**data)

    def _format_report_as_text(self, report: PhysicianReport) -> str:
        """Format report as plain text."""
        lines = [
            "=" * 80,
            f"PHYSICIAN REPORT - {report.report_id}",
            "=" * 80,
            f"Patient ID: {report.patient_id}",
            f"Generated: {report.report_generated_at}",
            "",
            "ASSESSMENT SUMMARY",
            "-" * 80,
        ]

        assessment = report.assessment

        # Suspected conditions
        lines.append("Suspected Conditions:")
        for i, condition in enumerate(assessment.suspected_conditions, 1):
            confidence = condition.get('confidence', 0)
            lines.append(f"  {i}. {condition.get('name', 'Unknown')} (Confidence: {confidence:.1%})")

        lines.extend([
            "",
            "Risk Factors:",
        ])

        for risk in assessment.risk_factors:
            lines.append(f"  • {risk}")

        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 80,
        ])

        for i, rec in enumerate(report.evidence_based_recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.extend([
            "",
            "SIMILAR CASES",
            "-" * 80,
            report.similar_cases_summary,
            "",
            "FOLLOW-UP ACTIONS",
            "-" * 80,
        ])

        for i, action in enumerate(report.follow_up_actions, 1):
            lines.append(f"{i}. {action}")

        lines.extend([
            "",
            "=" * 80
        ])

        return "\n".join(lines)
