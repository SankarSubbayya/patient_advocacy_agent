#!/usr/bin/env python
"""
Plot training losses from the training history.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training history
history_path = Path('/home/sankar/models/embedder_labeled_full/final/training_history.json')

if history_path.exists():
    with open(history_path, 'r') as f:
        history = json.load(f)
    print(f"Loaded history from: {history_path}")
else:
    print(f"History file not found: {history_path}")
    exit(1)

# Extract data
train_losses = history['train_losses']
val_losses = history['val_losses']
num_epochs = len(train_losses)
epochs = list(range(1, num_epochs + 1))

# Print statistics
print("\nTraining Statistics:")
print("="*60)
print(f"Epochs trained: {num_epochs} (early stopped from 20)")
print(f"Learning rate: {history['config']['learning_rate']}")
print(f"Batch size: {history['config']['batch_size']}")
print(f"Number of conditions: {history['config']['num_conditions']}")
print()

print("Loss values:")
print("-"*40)
print(f"Initial train loss: {train_losses[0]:.6f}")
print(f"Final train loss: {train_losses[-1]:.6f}")
print(f"Train loss change: {train_losses[0] - train_losses[-1]:.6f}")
print()
print(f"Initial val loss: {val_losses[0]:.6f}")
print(f"Final val loss: {val_losses[-1]:.6f}")
print(f"Best val loss: {history['best_val_loss']:.6f}")
print(f"Val loss change: {val_losses[0] - val_losses[-1]:.6f}")

# Create figure with multiple subplots
fig = plt.figure(figsize=(15, 10))

# 1. Main loss plot
ax1 = plt.subplot(2, 3, 1)
ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value annotations
for i in [0, -1]:
    ax1.annotate(f'{train_losses[i]:.4f}',
                xy=(epochs[i], train_losses[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax1.annotate(f'{val_losses[i]:.4f}',
                xy=(epochs[i], val_losses[i]),
                xytext=(5, -15), textcoords='offset points', fontsize=8)

# 2. Zoomed in plot (to see small changes)
ax2 = plt.subplot(2, 3, 2)
ax2.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
ax2.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss (Zoomed View)')
ax2.legend()
ax2.grid(True, alpha=0.3)
# Set y-axis limits for zoom
y_min = min(min(train_losses), min(val_losses)) - 0.001
y_max = max(max(train_losses), max(val_losses)) + 0.001
ax2.set_ylim([y_min, y_max])

# 3. Loss difference from initial
ax3 = plt.subplot(2, 3, 3)
train_diff = [train_losses[0] - l for l in train_losses]
val_diff = [val_losses[0] - l for l in val_losses]
ax3.plot(epochs, train_diff, 'b-o', label='Train Improvement', linewidth=2, markersize=6)
ax3.plot(epochs, val_diff, 'r-s', label='Val Improvement', linewidth=2, markersize=6)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss Reduction from Initial')
ax3.set_title('Loss Improvement Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 4. Loss change per epoch
ax4 = plt.subplot(2, 3, 4)
train_changes = [0] + [train_losses[i] - train_losses[i-1] for i in range(1, len(train_losses))]
val_changes = [0] + [val_losses[i] - val_losses[i-1] for i in range(1, len(val_losses))]
ax4.bar(np.array(epochs) - 0.2, train_changes, width=0.4, label='Train Change', alpha=0.7)
ax4.bar(np.array(epochs) + 0.2, val_changes, width=0.4, label='Val Change', alpha=0.7)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss Change from Previous Epoch')
ax4.set_title('Epoch-to-Epoch Loss Changes')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 5. Percentage change
ax5 = plt.subplot(2, 3, 5)
train_pct = [(train_losses[0] - l) / train_losses[0] * 100 for l in train_losses]
val_pct = [(val_losses[0] - l) / val_losses[0] * 100 for l in val_losses]
ax5.plot(epochs, train_pct, 'b-o', label='Train %', linewidth=2, markersize=6)
ax5.plot(epochs, val_pct, 'r-s', label='Val %', linewidth=2, markersize=6)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('% Improvement')
ax5.set_title('Percentage Improvement from Initial')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 6. Analysis text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
analysis_text = f"""
TRAINING ANALYSIS

Problem Detected:
✗ Loss is nearly flat (stuck ~3.455)
✗ Minimal learning occurred
✗ Early stopping at epoch {num_epochs}

Loss Statistics:
• Train: {train_losses[0]:.4f} → {train_losses[-1]:.4f}
• Val: {val_losses[0]:.4f} → {val_losses[-1]:.4f}
• Total change: {abs(train_losses[0] - train_losses[-1]):.6f}

This indicates:
1. Model is not learning effectively
2. Loss stuck at initialization value
3. Possible issues:
   - Learning rate too small
   - Data/label mismatch
   - Model architecture issue

Recommendation:
Try higher learning rate (1e-3)
or different optimizer (Adam)
"""
ax6.text(0.1, 0.9, analysis_text, fontsize=10, verticalalignment='top',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(f'Training Loss Analysis - {history["config"]["num_conditions"]} Conditions', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
output_path = Path('/home/sankar/patient_advocacy_agent/loss_analysis_plot.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_path}")

# Also save to model directory
model_plot_path = Path('/home/sankar/models/embedder_labeled_full/final/loss_analysis.png')
plt.savefig(model_plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Also saved to: {model_plot_path}")

plt.show()

# Final verdict
print("\n" + "="*60)
print("VERDICT: Training Failed to Learn")
print("="*60)
print("Despite having 30 different labeled conditions, the model")
print("is still stuck at the same loss value (~3.455).")
print("\nThis suggests the synthetic labels might not be providing")
print("enough learning signal, or there's an issue with how the")
print("text descriptions are being processed.")
print("\nNext steps to try:")
print("1. Use real labeled data (download matching images)")
print("2. Try much higher learning rate (1e-3)")
print("3. Check if text encoder is working properly")
print("4. Try different loss function or model architecture")