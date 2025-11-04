"""Patient Advocacy Agent with Claude API integration."""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
from anthropic import Anthropic

from .agent import MedGeminiAgent, PatientCase, AssessmentResult

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return {}
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return {}


class ClaudePatientAgent(MedGeminiAgent):
    """Enhanced Patient Advocacy Agent using Claude API for natural language generation."""
    
    def __init__(
        self,
        embedder,
        rag_pipeline,
        clustering_index,
        api_key: str = None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize Claude-powered patient agent.
        
        Args:
            embedder: SigLIPEmbedder instance
            rag_pipeline: RAGPipeline instance
            clustering_index: SimilarityIndex instance
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            config_path: Path to config file
        """
        super().__init__(embedder, rag_pipeline, clustering_index)
        
        # Load config
        config = load_config(config_path)
        claude_config = config.get('claude', {})
        
        # Initialize Claude
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = claude_config.get('model', 'claude-3-5-sonnet-20241022')
        self.max_tokens = claude_config.get('max_tokens', 1024)
        self.temperature = claude_config.get('temperature', 0.7)
        
        logger.info(f"Claude agent initialized with model: {self.model}")
    
    def generate_patient_explanation(
        self,
        patient_case: PatientCase,
        assessment: AssessmentResult,
        style: str = "empathetic"
    ) -> str:
        """
        Generate a patient-friendly explanation using Claude.
        
        Args:
            patient_case: Original patient case
            assessment: Assessment result
            style: Response style ('empathetic', 'educational', 'concise')
            
        Returns:
            Natural language explanation for the patient
        """
        # Build context from assessment
        context = self._build_patient_context(patient_case, assessment)
        
        # Create system prompt based on style
        system_prompt = self._get_system_prompt(style)
        
        # Create user message
        user_message = f"""Please provide a patient-friendly explanation based on this medical assessment:

{context}

The patient asked: "What might be causing my symptoms and what should I do?"

Please provide a helpful, empathetic response that:
1. Explains what their symptoms might indicate (in simple terms)
2. Describes what similar patients have experienced
3. Suggests next steps and when to seek care
4. Reassures while being medically accurate

Remember: This is educational information, not a diagnosis."""
        
        # Call Claude
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._fallback_explanation(patient_case, assessment)
    
    def generate_physician_summary(
        self,
        patient_case: PatientCase,
        assessment: AssessmentResult
    ) -> str:
        """
        Generate a professional physician summary using Claude.
        
        Args:
            patient_case: Patient case
            assessment: Assessment result
            
        Returns:
            Professional clinical summary
        """
        context = self._build_clinical_context(patient_case, assessment)
        
        system_prompt = """You are a clinical documentation assistant. Generate concise, professional medical summaries for physicians."""
        
        user_message = f"""Generate a clinical summary for this patient assessment:

{context}

Format the summary with:
## Chief Complaint
## Clinical Findings
## Differential Diagnosis
## Recommended Management
## Follow-up Plan

Use professional medical terminology."""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.3,  # Lower temperature for clinical docs
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._fallback_clinical_summary(patient_case, assessment)
    
    def answer_patient_question(
        self,
        question: str,
        patient_case: PatientCase,
        assessment: AssessmentResult
    ) -> str:
        """
        Answer a specific patient question using Claude with context.
        
        Args:
            question: Patient's question
            patient_case: Patient case
            assessment: Assessment result
            
        Returns:
            Claude's answer
        """
        context = self._build_patient_context(patient_case, assessment)
        
        system_prompt = self._get_system_prompt("empathetic")
        
        user_message = f"""Given this patient's medical context:

{context}

The patient asks: "{question}"

Please provide a helpful, accurate response. Be empathetic but clear about the importance of professional medical care."""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return "I apologize, but I'm having trouble generating a response. Please consult with a healthcare professional for medical advice."
    
    def _build_patient_context(self, patient_case: PatientCase, assessment: AssessmentResult) -> str:
        """Build context string for patient-facing responses."""
        context_parts = []
        
        # Patient info
        context_parts.append(f"Patient: {patient_case.age} year old {patient_case.gender}")
        context_parts.append(f"Symptoms: {', '.join(patient_case.symptoms)}")
        if patient_case.symptom_onset:
            context_parts.append(f"Duration: {patient_case.symptom_onset}")
        
        # Assessment findings
        if assessment.suspected_conditions:
            context_parts.append("\nPossible conditions:")
            for cond in assessment.suspected_conditions[:3]:
                context_parts.append(f"- {cond['name'].title()} (confidence: {cond.get('confidence', 0):.0%})")
        
        # Similar cases
        if assessment.similar_cases:
            context_parts.append(f"\nFound {len(assessment.similar_cases)} similar cases in database")
            for i, case in enumerate(assessment.similar_cases[:2], 1):
                context_parts.append(
                    f"  Case {i}: {case.get('condition', 'Unknown')} - "
                    f"Similar: {case.get('similarity_score', 0):.0%}"
                )
        
        # Medical knowledge
        if assessment.knowledge_summary:
            context_parts.append(f"\nMedical Knowledge:\n{assessment.knowledge_summary}")
        
        return "\n".join(context_parts)
    
    def _build_clinical_context(self, patient_case: PatientCase, assessment: AssessmentResult) -> str:
        """Build context string for clinical summaries."""
        context_parts = []
        
        # Demographics
        context_parts.append(f"Patient: {patient_case.age}yo {patient_case.gender}, ID: {patient_case.patient_id}")
        
        # Presentation
        context_parts.append(f"\nChief Complaint: {', '.join(patient_case.symptoms)}")
        if patient_case.symptom_onset:
            context_parts.append(f"Onset: {patient_case.symptom_onset}")
        if patient_case.patient_notes:
            context_parts.append(f"Notes: {patient_case.patient_notes}")
        
        # Clinical findings
        context_parts.append(f"\nDifferential Diagnosis:")
        for cond in assessment.suspected_conditions:
            context_parts.append(
                f"- {cond['name'].title()}: {cond.get('confidence', 0):.0%} confidence"
            )
        
        # Risk factors
        if assessment.risk_factors:
            context_parts.append(f"\nRisk Factors: {', '.join(assessment.risk_factors)}")
        
        # Recommendations
        if assessment.recommendations:
            context_parts.append(f"\nRecommendations:")
            for rec in assessment.recommendations:
                context_parts.append(f"- {rec}")
        
        return "\n".join(context_parts)
    
    def _get_system_prompt(self, style: str) -> str:
        """Get system prompt based on response style."""
        prompts = {
            "empathetic": """You are a compassionate medical advocacy assistant helping patients understand their skin conditions. 

Your role:
- Provide educational information (NOT medical diagnosis)
- Be empathetic and supportive
- Explain medical concepts in simple terms
- Encourage professional medical consultation
- Be honest about limitations

Important:
- You are NOT a doctor
- Always recommend consulting healthcare professionals
- Base responses on provided medical context
- Use warm, reassuring language""",

            "educational": """You are a medical education assistant specializing in dermatology.

Your role:
- Educate patients about skin conditions
- Explain symptoms, causes, and treatments clearly
- Use accessible language without oversimplifying
- Cite evidence-based information
- Emphasize the importance of professional care

Guidelines:
- Be factual and educational
- Avoid making diagnoses
- Encourage questions
- Promote health literacy""",

            "concise": """You are a medical information assistant providing brief, clear answers.

Your role:
- Give concise, accurate information
- Focus on key points
- Use bullet points when appropriate
- Direct patients to proper care

Keep responses focused and actionable."""
        }
        
        return prompts.get(style, prompts["empathetic"])
    
    def _fallback_explanation(self, patient_case: PatientCase, assessment: AssessmentResult) -> str:
        """Fallback explanation if Claude API fails."""
        if not assessment.suspected_conditions:
            return "I recommend consulting with a healthcare professional to evaluate your symptoms."
        
        top_condition = assessment.suspected_conditions[0]['name'].replace('_', ' ').title()
        
        return f"""Based on your symptoms ({', '.join(patient_case.symptoms)}), this may be related to {top_condition}.

I found {len(assessment.similar_cases)} similar cases in our database.

Recommendations:
{chr(10).join('- ' + rec for rec in assessment.recommendations[:3])}

Important: This is educational information only. Please consult with a healthcare professional for proper diagnosis and treatment."""
    
    def _fallback_clinical_summary(self, patient_case: PatientCase, assessment: AssessmentResult) -> str:
        """Fallback clinical summary if Claude API fails."""
        return f"""## Patient Assessment

**Demographics:** {patient_case.age}yo {patient_case.gender}
**Symptoms:** {', '.join(patient_case.symptoms)}

**Differential Diagnosis:**
{chr(10).join(f'{i+1}. {c["name"].title()}' for i, c in enumerate(assessment.suspected_conditions))}

**Recommendations:**
{chr(10).join(f'- {rec}' for rec in assessment.recommendations)}

**Follow-up:** Recommend clinical evaluation within 2-4 weeks.
"""


