import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_seld_output(predicted_sed, predicted_sel, true_sed, true_sel, classes, filename='seld_output.png'):
    """
    Visualize SED and SEL results along with ground truth and save the plot to a file.
    
    Parameters:
    - predicted_sed: Tensor of shape [B, T, S, C] with predicted SED logits
    - predicted_sel: Tensor of shape [B, T, S, C, 3] with predicted SEL coordinates
    - true_sed: Tensor of shape [B, T, S, C] with ground truth SED labels (one-hot encoded)
    - true_sel: Tensor of shape [B, T, S, C, 3] with ground truth SEL coordinates
    - classes: List of class names
    - filename: Filename for saving the plot
    """
    B, T, S, C = predicted_sed.shape

    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    
    for b in range(B):
        for s in range(S):
            for c in range(C):
                # Plot SED results
                axs[0].plot(range(T), torch.sigmoid(predicted_sed[b, :, s, c]).cpu(), label=f'Pred {classes[c]}', alpha=0.6)
                axs[0].plot(range(T), true_sed[b, :, s, c].cpu(), label=f'True {classes[c]}', linestyle='dashed', alpha=0.6)
                
                # Plot SEL results
                predicted_sel_coords = predicted_sel[b, :, s, c, :].cpu().numpy()
                true_sel_coords = true_sel[b, :, s, c, :].cpu().numpy()
                
                # Calculate the DoA as angles for better visualization
                predicted_angles = np.rad2deg(np.arctan2(predicted_sel_coords[:, 1], predicted_sel_coords[:, 0]))
                true_angles = np.rad2deg(np.arctan2(true_sel_coords[:, 1], true_sel_coords[:, 0]))
                
                axs[1].plot(range(T), predicted_angles, label=f'Pred {classes[c]}', alpha=0.6)
                axs[1].plot(range(T), true_angles, label=f'True {classes[c]}', linestyle='dashed', alpha=0.6)
                

    axs[0].set_title('Sound Event Detection (SED)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Probability')
    axs[0].legend(loc='upper right')

    axs[1].set_title('Sound Event Localization (SEL)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Angle (degrees)')
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()