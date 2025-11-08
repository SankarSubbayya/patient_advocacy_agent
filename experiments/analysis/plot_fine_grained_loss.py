#!/usr/bin/env python
"""
Plot the training and validation loss curves for the fine-grained SigLIP model.
"""

import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read the training log
log_file = Path('fine_grained_training.log')
print(f"Reading log file: {log_file}")

# Extract loss values from log
train_losses = []
eval_losses = []
epochs_for_eval = []

with open(log_file, 'r') as f:
    content = f.read()

    # Extract training losses (from logging steps)
    train_pattern = r"'loss': ([\d.]+)"
    train_matches = re.findall(train_pattern, content)
    train_losses = [float(loss) for loss in train_matches]

    # Extract eval losses (from eval results)
    eval_pattern = r"'eval_loss': ([\d.]+).*?'epoch': ([\d.]+)"
    eval_matches = re.findall(eval_pattern, content)
    for loss, epoch in eval_matches:
        eval_losses.append(float(loss))
        epochs_for_eval.append(float(epoch))

print(f"Found {len(train_losses)} training loss points")
print(f"Found {len(eval_losses)} evaluation loss points")

# Also load the summary for final metrics
summary_file = Path('/home/sankar/models/siglip_fine_grained/training_summary.json')
if summary_file.exists():
    with open(summary_file, 'r') as f:
        summary = json.load(f)
        final_train_loss = summary['final_loss']
        final_test_loss = summary['test_loss']
        print(f"\nFinal metrics from summary:")
        print(f"  Final training loss: {final_train_loss:.4f}")
        print(f"  Final test loss: {final_test_loss:.4f}")

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Training loss over iterations
if train_losses:
    iterations = np.arange(1, len(train_losses) + 1)
    ax1.plot(iterations, train_losses, 'b-', alpha=0.7, linewidth=1.5, label='Training Loss')

    # Add smoothed line
    if len(train_losses) > 10:
        window_size = min(20, len(train_losses) // 10)
        smoothed = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        smooth_x = iterations[window_size//2:len(smoothed)+window_size//2]
        ax1.plot(smooth_x, smoothed, 'b-', linewidth=2.5, label='Smoothed', alpha=0.9)

    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Fine-Grained Contrastive Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add annotations
    ax1.axhline(y=train_losses[0], color='g', linestyle='--', alpha=0.3, label=f'Initial: {train_losses[0]:.2f}')
    ax1.axhline(y=train_losses[-1], color='r', linestyle='--', alpha=0.3, label=f'Final: {train_losses[-1]:.2f}')

    # Calculate and show reduction
    reduction = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    ax1.text(0.02, 0.98, f'Loss Reduction: {reduction:.1f}%',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Training vs Validation loss by epoch
if eval_losses and epochs_for_eval:
    ax2.plot(epochs_for_eval, eval_losses, 'ro-', linewidth=2, markersize=8, label='Validation Loss')

    # Add training loss at eval points if we can estimate them
    if train_losses:
        # Estimate training loss at each epoch
        total_epochs = 25
        steps_per_epoch = len(train_losses) / total_epochs
        train_at_epochs = []
        for epoch in epochs_for_eval:
            step_idx = min(int(epoch * steps_per_epoch) - 1, len(train_losses) - 1)
            if step_idx >= 0:
                train_at_epochs.append(train_losses[step_idx])

        if train_at_epochs:
            ax2.plot(epochs_for_eval[:len(train_at_epochs)], train_at_epochs, 'bo-',
                    linewidth=2, markersize=8, label='Training Loss')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Check for overfitting
    if len(eval_losses) > 1:
        if eval_losses[-1] > min(eval_losses) * 1.1:
            ax2.text(0.98, 0.98, '⚠ Possible Overfitting',
                    transform=ax2.transAxes, fontsize=11,
                    horizontalalignment='right',
                    verticalalignment='top',
                    color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Add overall title
fig.suptitle('Fine-Grained SigLIP Model Training Analysis\n66 Condition Classes',
             fontsize=16, fontweight='bold')

# Add model comparison text
comparison_text = """Model Comparison (Retrieval Accuracy):
• Fine-Grained: 20%
• Basic Contrastive: 20%
• Hierarchical: 13%"""

fig.text(0.02, 0.02, comparison_text, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()

# Save the plot
output_file = 'fine_grained_loss_plot.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Also create a detailed metrics plot
fig2, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 3: Loss reduction over time
if train_losses:
    # Calculate percentage reduction at each step
    reductions = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
    ax3.plot(iterations, reductions, 'g-', linewidth=2)
    ax3.fill_between(iterations, 0, reductions, alpha=0.3, color='green')
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Loss Reduction (%)', fontsize=12)
    ax3.set_title('Cumulative Loss Reduction', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=50, color='r', linestyle='--', alpha=0.3, label='50% Reduction')
    ax3.legend()

# Plot 4: Learning rate schedule (estimated)
if train_losses:
    warmup_steps = 500
    initial_lr = 1e-5

    # Estimate learning rate schedule
    steps = np.arange(1, len(train_losses) + 1)
    lr_schedule = []
    for step in steps:
        if step <= warmup_steps:
            lr = initial_lr * step / warmup_steps
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / (len(train_losses) - warmup_steps)
            lr = initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        lr_schedule.append(lr)

    ax4.plot(steps, lr_schedule, 'orange', linewidth=2)
    ax4.set_xlabel('Training Steps', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.set_title('Learning Rate Schedule (Estimated)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Plot 5: Gradient of loss (rate of improvement)
if train_losses and len(train_losses) > 1:
    gradients = np.diff(train_losses)
    grad_steps = iterations[1:]

    ax5.plot(grad_steps, gradients, 'b-', alpha=0.5, linewidth=1)

    # Add smoothed gradient
    if len(gradients) > 10:
        window_size = min(20, len(gradients) // 10)
        smoothed_grad = np.convolve(gradients, np.ones(window_size)/window_size, mode='valid')
        smooth_grad_x = grad_steps[window_size//2:len(smoothed_grad)+window_size//2]
        ax5.plot(smooth_grad_x, smoothed_grad, 'r-', linewidth=2, label='Smoothed')

    ax5.axhline(y=0, color='g', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Training Steps', fontsize=12)
    ax5.set_ylabel('Loss Gradient (Δ Loss)', fontsize=12)
    ax5.set_title('Rate of Loss Improvement', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

# Plot 6: Model performance comparison
models = ['Fine-Grained\n(66 classes)', 'Basic Contrastive\n(16 classes)', 'Hierarchical\n(16+66 classes)']
accuracies = [20, 20, 13]
colors = ['green', 'blue', 'orange']

bars = ax6.bar(models, accuracies, color=colors, alpha=0.7)
ax6.axhline(y=10, color='r', linestyle='--', alpha=0.3, label='Random Baseline (10%)')
ax6.set_ylabel('Retrieval Accuracy (%)', fontsize=12)
ax6.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax6.set_ylim([0, 25])

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{acc}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

fig2.suptitle('Fine-Grained SigLIP Training - Detailed Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save the detailed plot
output_file2 = 'fine_grained_detailed_analysis.png'
fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"Detailed plot saved to: {output_file2}")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("TRAINING STATISTICS")
print("="*60)
if train_losses:
    print(f"Initial Loss: {train_losses[0]:.4f}")
    print(f"Final Loss: {train_losses[-1]:.4f}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Average Loss: {np.mean(train_losses):.4f}")
    print(f"Std Dev: {np.std(train_losses):.4f}")
    print(f"Min Loss: {min(train_losses):.4f}")
    print(f"Max Loss: {max(train_losses):.4f}")

if eval_losses:
    print(f"\nValidation Metrics:")
    print(f"Final Validation Loss: {eval_losses[-1]:.4f}")
    print(f"Best Validation Loss: {min(eval_losses):.4f}")
    print(f"Average Validation Loss: {np.mean(eval_losses):.4f}")