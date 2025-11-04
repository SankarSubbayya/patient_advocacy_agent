#!/bin/bash
# Install Claude API dependencies

echo "=================================="
echo "Installing Claude API Dependencies"
echo "=================================="
echo ""

# Install dependencies
echo "Installing anthropic and pyyaml..."
uv pip install anthropic pyyaml

echo ""
echo "✓ Installation complete!"
echo ""
echo "=================================="
echo "Next Steps"
echo "=================================="
echo ""
echo "1. Get your API key from: https://console.anthropic.com/"
echo ""
echo "2. Set your API key:"
echo "   export ANTHROPIC_API_KEY='your-api-key-here'"
echo ""
echo "3. Run the example:"
echo "   uv run python example_claude_usage.py"
echo ""
echo "📖 See CLAUDE_SETUP.md for complete guide"
echo ""


