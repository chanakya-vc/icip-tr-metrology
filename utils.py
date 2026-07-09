
import torch
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

plt.switch_backend('agg')  # for servers not supporting display
import os
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_train_val_split(full_train_dataset, val_fraction=0.2, batch_size=32):
    """
    Splits a PyTorch dataset into training and validation DataLoaders.

    Args:
        full_train_dataset (Dataset): The instantiated SEMSimulatorDataset.
        val_fraction (float): The fraction of data to use for validation (e.g., 0.2 for 20%).
        batch_size (int): Batch size for the DataLoaders.

    Returns:
        train_loader, val_loader
    """
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_fraction)
    train_size = total_size - val_size

    # Use a fixed generator seed so your split is reproducible across runs!
    # If you don't do this, your model might evaluate on data it trained on in a previous run.
    generator = torch.Generator().manual_seed(42)

    # Perform the split
    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Create DataLoaders
    # Note: We shuffle the train_loader, but validation doesn't need shuffling
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    return train_loader, val_loader

def plot_losses(loss_dir,running_train_loss, running_val_loss, train_epoch_loss, val_epoch_loss, epoch):
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle('loss trends', fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text('epoch train loss VS #epochs')
    ax1.set_xlabel('#epochs')
    ax1.set_ylabel('epoch train loss')
    ax1.plot(train_epoch_loss)

    ax2.title.set_text('epoch val loss VS #epochs')
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('epoch val loss')
    ax2.plot(val_epoch_loss)

    ax3.title.set_text('batch train loss VS #batches')
    ax3.set_xlabel('#batches')
    ax3.set_ylabel('batch train loss')
    ax3.plot(running_train_loss)

    ax4.title.set_text('batch val loss VS #batches')
    ax4.set_xlabel('#batches')
    ax4.set_ylabel('batch val loss')
    ax4.plot(running_val_loss)

    plt.savefig(os.path.join(loss_dir, 'losses_{}.png'.format(str(epoch + 1).zfill(2))))
    plt.close(fig)


def conv_estimator(data,dose):
    return np.nansum(data)/dose




def ctml_estimator(window_data, lam, tol=1e-7, max_iter=50):
    # Convert inputs to float64 arrays to avoid numerical truncation
    Y = np.nansum(window_data, axis=0)
    M_tilde = np.sum(~np.isnan(window_data), axis=0)
    Y = np.asarray(Y, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)

    # Pre-calculate the base denominator to avoid division by zero
    denominator_base = M_tilde + lam

    # Step 1: Initialize with a smart first guess
    # Using the standard MLE (Y / total_dose) as the starting point guarantees rapid convergence
    eta = np.zeros_like(Y)
    mask = denominator_base > 0
    eta[mask] = Y[mask] / denominator_base[mask]

    # Step 2: Fixed-Point Iteration
    for _ in range(max_iter):
        # Calculate the new denominator using the current eta estimate
        denominator = M_tilde + lam * np.exp(-eta)

        # Calculate the new eta (handling any potential division by zero gracefully)
        eta_next = np.zeros_like(Y)
        valid = denominator > 0
        eta_next[valid] = Y[valid] / denominator[valid]

        # Check if the largest update across the entire array is smaller than our tolerance
        if np.max(np.abs(eta_next - eta)) < tol:
            return eta_next

        eta = eta_next

    # If it hits max_iter without fully converging, it returns the best current estimate
    print(eta)
    return eta

