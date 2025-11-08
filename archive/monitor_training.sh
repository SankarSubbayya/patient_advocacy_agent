#!/bin/bash
# Monitor Hugging Face training progress

echo "Monitoring SigLIP fine-tuning progress..."
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo ""

while true; do
    clear
    echo "================================================================================"
    echo "SIGLIP TRAINING PROGRESS"
    echo "================================================================================"
    echo ""

    # Show last 30 lines of log
    tail -30 training_log_hf.txt

    echo ""
    echo "================================================================================"
    echo "Refreshing every 10 seconds... (Ctrl+C to stop monitoring)"
    echo "================================================================================"

    sleep 10
done
