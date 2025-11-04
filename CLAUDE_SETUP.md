# Claude API Integration Guide

This guide shows you how to use Claude API with your Patient Advocacy Agent.

## 📋 Prerequisites

1. **Trained Model**: You need to have trained the embedder first
2. **Anthropic API Key**: Get from https://console.anthropic.com/
3. **Python Package**: `anthropic` library

## 🚀 Quick Setup

### Step 1: Install Anthropic Library

```bash
cd /home/sankar/patient_advocacy_agent
uv pip install anthropic
```

### Step 2: Get Your API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-ant-...`)

### Step 3: Set Environment Variable

```bash
# For current session
export ANTHROPIC_API_KEY='your-api-key-here'

# Or add to your ~/.bashrc or ~/.zshrc for persistence
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Verify Setup

```bash
# Test connection
uv run python -c "from anthropic import Anthropic; client = Anthropic(); print('✓ Claude API connected')"
```

## 💡 Usage

### Basic Usage

```python
from patient_advocacy_agent.claude_agent import ClaudePatientAgent
from patient_advocacy_agent.agent import PatientCase

# Initialize agent (loads from config.yaml)
agent = ClaudePatientAgent(
    embedder=your_embedder,
    rag_pipeline=your_rag,
    clustering_index=your_index
)

# Create patient case
patient = PatientCase(
    patient_id="PAT-001",
    age=32,
    gender="Female",
    symptoms=["red itchy patches", "dry skin"],
    symptom_onset="3 months ago"
)

# Assess patient
assessment = agent.assess_patient(patient_case=patient)

# Generate patient-friendly explanation using Claude
explanation = agent.generate_patient_explanation(
    patient_case=patient,
    assessment=assessment,
    style="empathetic"  # or "educational", "concise"
)

print(explanation)
```

### Answer Patient Questions

```python
# Patient asks a follow-up question
answer = agent.answer_patient_question(
    question="Should I see a doctor?",
    patient_case=patient,
    assessment=assessment
)

print(answer)
```

### Generate Physician Summary

```python
# Clinical summary for healthcare providers
summary = agent.generate_physician_summary(
    patient_case=patient,
    assessment=assessment
)

print(summary)
```

## 🎯 Complete Example

Run the complete example:

```bash
# Make sure you have:
# 1. Trained model at ./models/embedder/final/embedder.pt
# 2. Built index at ./models/similarity_index
# 3. Set ANTHROPIC_API_KEY environment variable

uv run python example_claude_usage.py
```

## ⚙️ Configuration

Edit `config.yaml` to customize Claude settings:

```yaml
claude:
  model: claude-3-5-sonnet-20241022  # Latest Claude model
  max_tokens: 1024                    # Response length
  temperature: 0.7                     # Creativity (0-1)
```

Available models:
- `claude-3-5-sonnet-20241022` - Best balance (recommended)
- `claude-3-opus-20240229` - Most capable
- `claude-3-haiku-20240307` - Fastest, cheapest

## 💰 Pricing

As of Nov 2024:
- **Claude 3.5 Sonnet**: $3 per 1M input tokens, $15 per 1M output tokens
- **Claude 3 Haiku**: $0.25 per 1M input tokens, $1.25 per 1M output tokens

Typical costs:
- Patient explanation: ~$0.001 - $0.005 per query
- 1000 patient queries: ~$1 - $5

## 🔒 Security

### Best Practices

1. **Never commit API keys to git**
   ```bash
   # Add to .gitignore
   echo "*.env" >> .gitignore
   echo ".anthropic_key" >> .gitignore
   ```

2. **Use environment variables** (not hardcoded)
   ```python
   # Good ✓
   api_key = os.environ.get("ANTHROPIC_API_KEY")
   
   # Bad ✗
   api_key = "sk-ant-12345..."
   ```

3. **Rotate keys regularly**
   - Regenerate keys every 90 days
   - Delete unused keys

4. **Monitor usage**
   - Check console.anthropic.com for usage
   - Set spending limits

## 🎨 Response Styles

The agent supports different response styles:

### Empathetic (Default)
- Warm, supportive tone
- Patient-friendly language
- Encourages questions

```python
explanation = agent.generate_patient_explanation(
    patient_case=patient,
    assessment=assessment,
    style="empathetic"
)
```

### Educational
- Informative and factual
- Promotes health literacy
- Evidence-based

```python
explanation = agent.generate_patient_explanation(
    patient_case=patient,
    assessment=assessment,
    style="educational"
)
```

### Concise
- Brief and direct
- Bullet points
- Action-oriented

```python
explanation = agent.generate_patient_explanation(
    patient_case=patient,
    assessment=assessment,
    style="concise"
)
```

## 🐛 Troubleshooting

### Issue: "anthropic not installed"
```bash
uv pip install anthropic
```

### Issue: "API key not found"
```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set it
export ANTHROPIC_API_KEY='your-key'
```

### Issue: "Rate limit exceeded"
- You've hit API rate limits
- Wait a few seconds and retry
- Consider upgrading your plan

### Issue: "Invalid API key"
- Double-check your key (should start with `sk-ant-`)
- Make sure it's not expired
- Generate a new key if needed

## 📚 Additional Resources

- [Anthropic API Docs](https://docs.anthropic.com/)
- [Claude Pricing](https://www.anthropic.com/pricing)
- [API Console](https://console.anthropic.com/)
- [Python SDK](https://github.com/anthropics/anthropic-sdk-python)

## 🔄 Fallback Behavior

If Claude API is unavailable (no key, rate limit, etc.), the agent automatically falls back to rule-based responses. This ensures your application continues working even without Claude.

```python
# Agent automatically handles fallback
try:
    response = agent.generate_patient_explanation(...)
except Exception:
    # Returns rule-based response
    response = agent._fallback_explanation(...)
```

## ✅ Next Steps

1. ✓ Set up API key
2. ✓ Install anthropic library
3. ⬜ Train your embedder model
4. ⬜ Build similarity index
5. ⬜ Run example_claude_usage.py
6. ⬜ Build web interface
7. ⬜ Deploy to production

---

**Questions?** Check the example files:
- `example_claude_usage.py` - Complete end-to-end example
- `claude_integration_example.py` - Simple API usage
- `src/patient_advocacy_agent/claude_agent.py` - Implementation


