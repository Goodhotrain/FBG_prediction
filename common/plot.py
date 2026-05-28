import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_convergence_analysis(train_loss, test_loss, test_acc, x_axis_data=None, 
                             smooth_weight=0.9, figsize=(9, 6), save_dir='.'):
    """
    Plot training convergence analysis charts.

    Args:
    train_loss: list of training losses
    test_loss: list of test losses
    test_acc: list of test accuracies
    x_axis_data: X-axis data (e.g., epoch numbers), auto-generated if None
    smooth_weight: smoothing weight, default 0.9
    figsize: figure size, default (9, 6)
    save_dir: save directory, default current directory
    """
    # Set up fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    def ema_smooth(values, weight=0.9):
        """Exponential moving average smoothing"""
        if len(values) == 0:
            return np.array(values)
        smoothed = []
        last = values[0]
        for v in values:
            last = last * weight + (1 - weight) * v
            smoothed.append(last)
        return np.array(smoothed)
    
    # Smooth data
    valid_loss_smooth = ema_smooth(test_loss, smooth_weight)
    train_loss_smooth = ema_smooth(train_loss, smooth_weight)
    
    # Generate X-axis data (if not provided)
    if x_axis_data is None:
        x_axis_data = list(range(len(train_loss)))
    
    # Save data to CSV
    pd.DataFrame({
        "test_loss": test_loss,
        "train_loss": train_loss
    }).to_csv(os.path.join(save_dir, "loss.csv"), index=False)
    
    pd.DataFrame({
        "test_acc": test_acc
    }).to_csv(os.path.join(save_dir, "acc.csv"), index=False)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot loss curves (primary Y-axis)
    ax = plt.gca()
    plt.plot(x_axis_data[:len(train_loss)], train_loss, color='#74b9ff', alpha=0.3, label='Train Loss')
    plt.plot(x_axis_data[:len(train_loss)], train_loss_smooth, color='#74b9ff', label='Train Loss (smoothed)')
    plt.plot(x_axis_data[:len(test_loss)], test_loss, color='orange', alpha=0.3, label='Valid Loss')
    plt.plot(x_axis_data[:len(test_loss)], valid_loss_smooth, color='orange', label='Valid Loss (smoothed)')
    
    # Plot accuracy curves (secondary Y-axis)
    ax2 = ax.twinx()
    ax2.plot(x_axis_data[:len(test_acc)], test_acc, linestyle='--', color='green', label='Valid Acc')
    
    # Set labels and title
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax2.set_ylabel('Accuracy', fontsize=16)
    ax.set_title('Convergence Analysis')
    
    # Merge legends
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    
    # Save figure
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'convergence.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f'Convergence figure saved to: {save_path}')
    return save_path

# Usage example:
# plot_convergence_analysis(
#     train_loss=self.train_loss,
#     test_loss=self.test_loss2, 
#     test_acc=self.test_loss,
#     x_axis_data=self.x_axis_data
# )