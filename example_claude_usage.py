#!/usr/bin/env python
"""
Complete example: Using the Patient Advocacy Agent with Claude API.

This demonstrates the full pipeline:
1. Load trained embedder and RAG system
2. Create Claude-powered agent
3. Assess a patient case
4. Generate patient-friendly explanations
5. Answer follow-up questions
"""

import os
import sys
from pathlib import Path
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from patient_advocacy_agent import (
    SigLIPEmbedder,
    SimilarityIndex,
    RAGPipeline
)
from patient_advocacy_agent.claude_agent import ClaudePatientAgent
from patient_advocacy_agent.agent import PatientCase


def load_models():
    """Load trained models and indices."""
    print("\n" + "="*80)
    print("Loading Models and Indices")
    print("="*80)
    
    # Paths (from config.yaml)
    embedder_path = Path("./models/embedder/final/embedder.pt")
    index_dir = Path("./models/similarity_index")
    rag_dir = Path("./models/rag_pipeline")
    
    # Check if models exist
    if not embedder_path.exists():
        print(f"✗ Embedder not found at {embedder_path}")
        print("Run: uv run python train_embedder.py")
        return None, None, None
    
    if not index_dir.exists():
        print(f"✗ Index not found at {index_dir}")
        print("Run: uv run python build_index.py")
        return None, None, None
    
    # Load embedder
    print("Loading embedder...")
    embedder = SigLIPEmbedder.load(embedder_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = embedder.to(device)
    print(f"✓ Embedder loaded on {device}")
    
    # Load similarity index
    print("Loading similarity index...")
    similarity_index = SimilarityIndex.load(index_dir)
    print(f"✓ Similarity index loaded")
    
    # Load RAG pipeline
    print("Loading RAG pipeline...")
    rag_pipeline = RAGPipeline.load(rag_dir)
    print(f"✓ RAG pipeline loaded")
    
    return embedder, similarity_index, rag_pipeline


def main():
    """Main example function."""
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n" + "="*80)
        print("⚠ ANTHROPIC_API_KEY not set")
        print("="*80)
        print("\nTo use Claude API, set your API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://console.anthropic.com/")
        print("\nFor now, the agent will use fallback responses (non-Claude).")
        print("="*80)
        input("\nPress Enter to continue with demo anyway...")
    
    # Load models
    embedder, similarity_index, rag_pipeline = load_models()
    
    if embedder is None:
        print("\n✗ Failed to load models. Please train the model first.")
        return 1
    
    # Initialize Claude agent
    print("\n" + "="*80)
    print("Initializing Claude Patient Agent")
    print("="*80)
    
    try:
        agent = ClaudePatientAgent(
            embedder=embedder,
            rag_pipeline=rag_pipeline,
            clustering_index=similarity_index
        )
        print("✓ Agent initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        return 1
    
    # Create example patient case
    print("\n" + "="*80)
    print("Example Patient Case")
    print("="*80)
    
    patient = PatientCase(
        patient_id="PAT-001",
        age=32,
        gender="Female",
        symptoms=[
            "red itchy patches on arms",
            "dry scaly skin",
            "worse in winter"
        ],
        symptom_onset="3 months ago",
        patient_notes="Gets worse at night, affecting sleep"
    )
    
    print(f"Patient ID: {patient.patient_id}")
    print(f"Age: {patient.age}, Gender: {patient.gender}")
    print(f"Symptoms: {', '.join(patient.symptoms)}")
    print(f"Onset: {patient.symptom_onset}")
    
    # Assess patient (without image for this example)
    print("\n" + "="*80)
    print("Performing Assessment")
    print("="*80)
    
    try:
        assessment = agent.assess_patient(
            patient_case=patient,
            image_tensor=None,  # You can add image processing here
            num_similar_cases=5
        )
        print("✓ Assessment complete")
        
        # Show assessment results
        print("\nSuspected Conditions:")
        for i, cond in enumerate(assessment.suspected_conditions, 1):
            print(f"  {i}. {cond['name'].title()}: {cond.get('confidence', 0):.0%} confidence")
        
        print(f"\nSimilar Cases Found: {len(assessment.similar_cases)}")
        print(f"Recommendations: {len(assessment.recommendations)}")
        
    except Exception as e:
        print(f"✗ Assessment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate patient explanation using Claude
    print("\n" + "="*80)
    print("Generating Patient Explanation (using Claude)")
    print("="*80)
    
    try:
        explanation = agent.generate_patient_explanation(
            patient_case=patient,
            assessment=assessment,
            style="empathetic"
        )
        
        print("\n" + "-"*80)
        print(explanation)
        print("-"*80)
        
    except Exception as e:
        print(f"Note: Claude API unavailable, using fallback: {e}")
        explanation = agent._fallback_explanation(patient, assessment)
        print("\n" + "-"*80)
        print(explanation)
        print("-"*80)
    
    # Answer a follow-up question
    print("\n" + "="*80)
    print("Answering Follow-up Question (using Claude)")
    print("="*80)
    
    question = "Should I be worried about this? When should I see a doctor?"
    print(f"\nPatient asks: \"{question}\"")
    print("\nClaude's response:")
    print("-"*80)
    
    try:
        answer = agent.answer_patient_question(
            question=question,
            patient_case=patient,
            assessment=assessment
        )
        print(answer)
        print("-"*80)
        
    except Exception as e:
        print(f"Note: Claude API unavailable: {e}")
        print("Please consult with a healthcare professional for medical advice.")
        print("-"*80)
    
    # Generate physician summary
    print("\n" + "="*80)
    print("Generating Physician Summary (using Claude)")
    print("="*80)
    
    try:
        summary = agent.generate_physician_summary(
            patient_case=patient,
            assessment=assessment
        )
        
        print("\n" + "-"*80)
        print(summary)
        print("-"*80)
        
    except Exception as e:
        print(f"Note: Claude API unavailable: {e}")
        summary = agent._fallback_clinical_summary(patient, assessment)
        print("\n" + "-"*80)
        print(summary)
        print("-"*80)
    
    print("\n" + "="*80)
    print("✓ Example Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Set ANTHROPIC_API_KEY to use Claude API")
    print("2. Add image processing for visual assessment")
    print("3. Build a web interface (see api.py)")
    print("4. Deploy as a service")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


