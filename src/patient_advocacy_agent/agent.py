"""MedGemini agent for patient assessment and physician report generation."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import numpy as np
from pydantic import BaseModel
from datetime import datetime

logger = logging.getLogger(__name__)


class PatientCase(BaseModel):
    """Patient case information."""
    patient_id: str
    age: int
    gender: str
    symptoms: List[str]
    symptom_onset: Optional[str] = None
    patient_notes: Optional[str] = None
    image_path: Optional[str] = None


class AssessmentResult(BaseModel):
    """Assessment result from the agent."""
    patient_id: str
    suspected_conditions: List[Dict[str, Any]]
    confidence_scores: List[float]
    similar_cases: List[Dict[str, Any]]
    recommendations: List[str]
    risk_factors: List[str]
    knowledge_summary: Optional[str] = None
    assessment_date: str
    agent_notes: Optional[str] = None


class PhysicianReport(BaseModel):
    """Formatted physician report."""
    report_id: str
    patient_id: str
    assessment: AssessmentResult
    similar_cases_summary: str
    evidence_based_recommendations: List[str]
    follow_up_actions: List[str]
    report_generated_at: str


class MedGeminiAgent:
    """AI agent for patient assessment using medical knowledge and similar cases."""

    def __init__(
        self,
        embedder,
        rag_pipeline,
        clustering_index
    ):
        """
        Initialize MedGemini agent.

        Args:
            embedder: SigLIPEmbedder instance for image processing
            rag_pipeline: RAGPipeline instance for knowledge retrieval
            clustering_index: SimilarityIndex instance for case matching
        """
        self.embedder = embedder
        self.rag_pipeline = rag_pipeline
        self.clustering_index = clustering_index

    def assess_patient(
        self,
        patient_case: PatientCase,
        image_tensor: Optional[np.ndarray] = None,
        num_similar_cases: int = 5,
        confidence_threshold: float = 0.3
    ) -> AssessmentResult:
        """
        Assess a patient case and generate recommendations.

        Args:
            patient_case: Patient case information
            image_tensor: Optional image tensor for embedding
            num_similar_cases: Number of similar cases to retrieve
            confidence_threshold: Minimum confidence threshold

        Returns:
            Assessment result with recommendations
        """
        logger.info(f"Assessing patient {patient_case.patient_id}")

        # Extract image embedding if provided
        image_embedding = None
        if image_tensor is not None:
            image_embedding = self.embedder.extract_image_features(
                image_tensor.unsqueeze(0).to(next(self.embedder.parameters()).device)
            ).detach().cpu().numpy()[0]

        # Generate suspected conditions based on symptoms
        suspected_conditions = self._identify_conditions(patient_case.symptoms)

        # Retrieve similar cases from database
        similar_cases = []
        if image_embedding is not None:
            similar_cases = self.clustering_index.search(image_embedding, k=num_similar_cases)
            similar_cases = [case.model_dump() for case in similar_cases]

        # Retrieve medical knowledge
        context = self.rag_pipeline.retrieve_context(
            condition=suspected_conditions[0]['name'] if suspected_conditions else 'unknown',
            symptoms=patient_case.symptoms,
            query_embedding=image_embedding,
            num_cases=num_similar_cases,
            num_knowledge=3
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            suspected_conditions,
            patient_case.symptoms,
            context.get('knowledge_docs', []),
            patient_case.age
        )

        # Identify risk factors
        risk_factors = self._identify_risk_factors(patient_case)

        # Filter conditions by confidence threshold
        filtered_conditions = [
            c for c in suspected_conditions
            if c.get('confidence', 0.5) >= confidence_threshold
        ]

        assessment = AssessmentResult(
            patient_id=patient_case.patient_id,
            suspected_conditions=filtered_conditions,
            confidence_scores=[c.get('confidence', 0.5) for c in filtered_conditions],
            similar_cases=similar_cases,
            recommendations=recommendations,
            risk_factors=risk_factors,
            knowledge_summary=self._summarize_knowledge(context.get('knowledge_docs', [])),
            assessment_date=datetime.now().isoformat(),
            agent_notes=f"Assessment for patient with symptoms: {', '.join(patient_case.symptoms)}"
        )

        logger.info(f"Assessment completed for patient {patient_case.patient_id}")
        return assessment

    def _identify_conditions(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """
        Identify likely skin conditions based on symptoms.

        This is a simplified heuristic-based approach. In production, this would use
        a trained classification model or lookup from medical database.

        Args:
            symptoms: List of symptoms

        Returns:
            List of suspected conditions with confidence scores
        """
        # Symptom-to-condition mappings (simplified for demo)
        symptom_patterns = {
            'eczema': {
                'symptoms': ['itching', 'redness', 'dryness', 'inflammation'],
                'base_confidence': 0.7
            },
            'psoriasis': {
                'symptoms': ['scaling', 'redness', 'plaques', 'itching'],
                'base_confidence': 0.7
            },
            'dermatitis': {
                'symptoms': ['rash', 'itching', 'redness', 'blistering'],
                'base_confidence': 0.6
            },
            'acne': {
                'symptoms': ['pimples', 'blackheads', 'inflammation', 'oily skin'],
                'base_confidence': 0.8
            },
            'fungal_infection': {
                'symptoms': ['itching', 'redness', 'scaling', 'discoloration'],
                'base_confidence': 0.6
            }
        }

        conditions = []
        symptoms_lower = [s.lower() for s in symptoms]

        for condition, pattern in symptom_patterns.items():
            matching_symptoms = sum(
                1 for sym in pattern['symptoms']
                if any(sym in patient_sym for patient_sym in symptoms_lower)
            )

            if matching_symptoms > 0:
                confidence = min(
                    1.0,
                    pattern['base_confidence'] * (matching_symptoms / len(pattern['symptoms']))
                )
                conditions.append({
                    'name': condition,
                    'confidence': confidence,
                    'matching_symptoms': matching_symptoms
                })

        # Sort by confidence
        conditions.sort(key=lambda x: x['confidence'], reverse=True)

        # Return top 3 conditions
        return conditions[:3] if conditions else [{'name': 'unknown', 'confidence': 0.3}]

    def _identify_risk_factors(self, patient_case: PatientCase) -> List[str]:
        """
        Identify risk factors for the patient.

        Args:
            patient_case: Patient case information

        Returns:
            List of risk factors
        """
        risk_factors = []

        # Age-based risk factors
        if patient_case.age < 5:
            risk_factors.append("Young age - increased sensitivity to skin conditions")
        elif patient_case.age > 60:
            risk_factors.append("Advanced age - may have compromised skin barrier function")

        # Gender-specific risk factors
        if patient_case.gender.lower() == 'female':
            risk_factors.append("Gender-specific: hormonal factors may influence skin condition")

        # Symptom-based risk factors
        symptoms_lower = [s.lower() for s in patient_case.symptoms]
        if any('infection' in s for s in symptoms_lower):
            risk_factors.append("Risk of secondary bacterial infection")
        if any('spreading' in s for s in symptoms_lower):
            risk_factors.append("Condition may be contagious")

        # Chronicity risk
        if patient_case.symptom_onset and 'months' in patient_case.symptom_onset.lower():
            risk_factors.append("Chronic presentation - may require long-term management")

        return risk_factors if risk_factors else ["No significant risk factors identified"]

    def _generate_recommendations(
        self,
        conditions: List[Dict[str, Any]],
        symptoms: List[str],
        knowledge_docs: List[Dict[str, Any]],
        patient_age: int
    ) -> List[str]:
        """
        Generate clinical recommendations based on assessment.

        Args:
            conditions: List of suspected conditions
            symptoms: Patient symptoms
            knowledge_docs: Retrieved medical knowledge documents
            patient_age: Patient age

        Returns:
            List of recommendations
        """
        recommendations = []

        # Base recommendations for top condition
        if conditions:
            top_condition = conditions[0]['name']

            condition_recommendations = {
                'eczema': [
                    'Prescribe topical corticosteroids for acute flare-ups',
                    'Recommend daily moisturizer with ceramides',
                    'Advise on skin barrier protection and avoiding irritants'
                ],
                'psoriasis': [
                    'Consider topical calcineurin inhibitors or vitamin D analogues',
                    'Recommend phototherapy if widespread',
                    'Monitor for systemic manifestations'
                ],
                'dermatitis': [
                    'Identify and eliminate triggering agents',
                    'Recommend hypoallergenic cleansers',
                    'Consider antihistamines for severe itching'
                ],
                'acne': [
                    'Consider topical retinoids or benzoyl peroxide',
                    'Recommend gentle cleansing and non-comedogenic products',
                    'Assess need for systemic antibiotics or isotretinoin'
                ],
                'fungal_infection': [
                    'Prescribe appropriate antifungal agent (topical or systemic)',
                    'Advise on hygiene and preventing spread',
                    'Monitor for treatment response'
                ]
            }

            recommendations.extend(
                condition_recommendations.get(top_condition, [
                    'Further clinical evaluation recommended',
                    'Consider dermatology referral for specialized assessment'
                ])
            )

        # Age-specific recommendations
        if patient_age < 12:
            recommendations.append("Use pediatric-formulated products due to young age")
        elif patient_age > 60:
            recommendations.append("Consider age-related changes in skin physiology")

        # General recommendations
        recommendations.extend([
            'Follow-up examination in 2-4 weeks to assess treatment response',
            'Document baseline severity for comparison',
            'Advise patient on expected timeline for improvement'
        ])

        return recommendations[:5]  # Return top 5 recommendations

    def _summarize_knowledge(self, knowledge_docs: List[Dict[str, Any]]) -> str:
        """
        Summarize retrieved knowledge documents.

        Args:
            knowledge_docs: Retrieved knowledge documents

        Returns:
            Summary string
        """
        if not knowledge_docs:
            return "No additional medical knowledge retrieved."

        summaries = []
        for doc in knowledge_docs[:3]:  # Top 3 documents
            content = doc.get('content', '')[:200]  # First 200 chars
            summaries.append(f"- {content}...")

        return "\n".join(summaries)

    def generate_physician_report(
        self,
        assessment: AssessmentResult,
        patient_case: PatientCase
    ) -> PhysicianReport:
        """
        Generate a formatted physician report.

        Args:
            assessment: Assessment result from agent
            patient_case: Original patient case information

        Returns:
            Formatted physician report
        """
        # Summarize similar cases
        similar_cases_summary = self._summarize_similar_cases(assessment.similar_cases)

        # Generate evidence-based recommendations
        evidence_based_recs = self._evidence_based_recommendations(
            assessment.suspected_conditions,
            assessment.risk_factors
        )

        # Determine follow-up actions
        follow_up = self._generate_follow_up_actions(assessment)

        report = PhysicianReport(
            report_id=f"PAA-{patient_case.patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            patient_id=patient_case.patient_id,
            assessment=assessment,
            similar_cases_summary=similar_cases_summary,
            evidence_based_recommendations=evidence_based_recs,
            follow_up_actions=follow_up,
            report_generated_at=datetime.now().isoformat()
        )

        return report

    def _summarize_similar_cases(self, similar_cases: List[Dict[str, Any]]) -> str:
        """Summarize similar cases for report."""
        if not similar_cases:
            return "No similar cases found in database."

        summary = "Similar Cases from Database:\n"
        for i, case in enumerate(similar_cases[:5], 1):
            summary += (
                f"{i}. Case {case.get('case_id', 'Unknown')} - "
                f"{case.get('condition', 'Unknown')} "
                f"(Similarity: {case.get('similarity_score', 0):.2%})\n"
            )

        return summary

    def _evidence_based_recommendations(
        self,
        conditions: List[Dict[str, Any]],
        risk_factors: List[str]
    ) -> List[str]:
        """Generate evidence-based recommendations."""
        recs = []

        if conditions:
            recs.append(
                f"Based on clinical presentation, primary differential: "
                f"{conditions[0].get('name', 'Unknown').title()}"
            )

        if "secondary infection" in str(risk_factors).lower():
            recs.append("Monitor closely for secondary infections")

        recs.append("Consider detailed history and examination findings")

        return recs

    def _generate_follow_up_actions(self, assessment: AssessmentResult) -> List[str]:
        """Generate follow-up actions."""
        actions = [
            "Schedule follow-up appointment in 2-4 weeks",
            "Provide patient education materials",
            "Document response to treatment"
        ]

        if assessment.confidence_scores and assessment.confidence_scores[0] < 0.6:
            actions.append("Consider specialist referral given diagnostic uncertainty")

        return actions

    def save(self, path: Path) -> None:
        """Save agent configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            'agent_type': 'MedGemini',
            'version': '1.0.0',
            'created_at': datetime.now().isoformat()
        }

        with open(path / "agent_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Agent configuration saved to {path}")
