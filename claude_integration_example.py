#!/usr/bin/env python
"""
Example: Integrating Claude API into Patient Advocacy Agent.

This shows how to use Claude for generating responses based on
retrieved medical cases and knowledge.
"""

import os
from typing import List, Dict
from anthropic import Anthropic


class ClaudeRAGAssistant:
    """Patient advocacy assistant using Claude API with RAG."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Claude assistant.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-3-5-sonnet-20241022"  # Latest model
    
    def generate_response(
        self,
        query: str,
        similar_cases: List[Dict],
        medical_knowledge: List[str],
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a response using Claude based on retrieved context.
        
        Args:
            query: User's question
            similar_cases: Similar cases from your FAISS index
            medical_knowledge: Retrieved medical information
            max_tokens: Maximum response length
            
        Returns:
            Claude's response
        """
        # Build context from retrieved information
        context_parts = []
        
        # Add similar cases
        if similar_cases:
            context_parts.append("## Similar Cases:")
            for i, case in enumerate(similar_cases[:3], 1):
                context_parts.append(f"\nCase {i}:")
                context_parts.append(f"- Condition: {case.get('condition', 'Unknown')}")
                context_parts.append(f"- Symptoms: {case.get('symptoms', 'N/A')}")
                context_parts.append(f"- Severity: {case.get('severity', 'N/A')}")
        
        # Add medical knowledge
        if medical_knowledge:
            context_parts.append("\n## Medical Knowledge:")
            for knowledge in medical_knowledge[:3]:
                context_parts.append(f"\n{knowledge}")
        
        context = "\n".join(context_parts)
        
        # Create system prompt
        system_prompt = """You are a helpful medical advocacy assistant. Your role is to:
1. Help patients understand their skin conditions
2. Provide educational information (NOT medical diagnosis)
3. Suggest when to seek professional medical care
4. Be empathetic and supportive

Important disclaimers:
- You are NOT a doctor and cannot diagnose
- Always recommend consulting healthcare professionals for medical advice
- Base your responses on the provided medical knowledge and similar cases"""
        
        # Create user message with context
        user_message = f"""Based on the following medical context, please help answer this question:

{context}

User Question: {query}

Please provide a helpful, empathetic response that educates the user while emphasizing the importance of professional medical consultation."""
        
        # Call Claude API
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        return message.content[0].text
    
    def stream_response(
        self,
        query: str,
        similar_cases: List[Dict],
        medical_knowledge: List[str]
    ):
        """
        Stream Claude's response for real-time display.
        
        Yields:
            Text chunks as they arrive
        """
        # Build context (same as above)
        context_parts = []
        
        if similar_cases:
            context_parts.append("## Similar Cases:")
            for i, case in enumerate(similar_cases[:3], 1):
                context_parts.append(f"\nCase {i}: {case.get('condition', 'Unknown')}")
        
        if medical_knowledge:
            context_parts.append("\n## Medical Knowledge:")
            for knowledge in medical_knowledge[:2]:
                context_parts.append(f"\n{knowledge}")
        
        context = "\n".join(context_parts)
        
        # Stream response
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"{context}\n\nQuestion: {query}"
            }]
        ) as stream:
            for text in stream.text_stream:
                yield text


# Example usage
def example_usage():
    """Example of using Claude with your RAG system."""
    
    # Initialize assistant
    assistant = ClaudeRAGAssistant()
    
    # Example: Retrieved information from your RAG system
    similar_cases = [
        {
            "condition": "eczema",
            "symptoms": "Red, itchy patches on arms",
            "severity": "moderate"
        },
        {
            "condition": "atopic dermatitis",
            "symptoms": "Dry, scaly skin with inflammation",
            "severity": "mild"
        }
    ]
    
    medical_knowledge = [
        "Eczema (atopic dermatitis) is a chronic inflammatory skin condition. "
        "Treatment focuses on moisturizing and reducing inflammation.",
        "Common triggers include dry weather, stress, and certain allergens."
    ]
    
    # User query
    query = "I have red, itchy patches on my arms. What could this be?"
    
    # Generate response
    print("Generating response...\n")
    response = assistant.generate_response(
        query=query,
        similar_cases=similar_cases,
        medical_knowledge=medical_knowledge
    )
    
    print("Claude's Response:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    
    # Stream response (alternative)
    print("\n\nStreaming response:")
    print("=" * 80)
    for chunk in assistant.stream_response(query, similar_cases, medical_knowledge):
        print(chunk, end="", flush=True)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠ Set your API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        print("\nOr pass it when initializing:")
        print("  assistant = ClaudeRAGAssistant(api_key='your-key')")
    else:
        example_usage()


