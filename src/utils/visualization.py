"""
Visualization functions for federated learning results
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_training_history(history, save_path=None):
    """
    Plot training history from federated learning
    
    Args:
        history: Dictionary with training history
        save_path: Path to save plots (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = history.get('round', list(range(len(history.get('baf_loss', [])))))
    
    # Loss plot
    if 'baf_loss' in history and 'ieee_loss' in history:
        axes[0].plot(rounds, history['baf_loss'], 'b-o', 
                    label='Bank Account Client', linewidth=2, markersize=4)
        axes[0].plot(rounds, history['ieee_loss'], 'r-s', 
                    label='Credit Card Client', linewidth=2, markersize=4)
        axes[0].set_xlabel('Federated Round', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'baf_accuracy' in history and 'ieee_accuracy' in history:
        axes[1].plot(rounds, history['baf_accuracy'], 'b-o', 
                    label='Bank Account Client', linewidth=2, markersize=4)
        axes[1].plot(rounds, history['ieee_accuracy'], 'r-s', 
                    label='Credit Card Client', linewidth=2, markersize=4)
        axes[1].set_xlabel('Federated Round', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/training_history.png", dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved: {save_path}/training_history.png")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['Legitimate', 'Fraud'], save_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f"{save_path}/roc_curve.png", dpi=300, bbox_inches='tight')
    
    plt.show()
