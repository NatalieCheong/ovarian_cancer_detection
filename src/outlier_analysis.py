import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import seaborn as sns

def analyze_outliers(model, dataloader, device, threshold=3.0):
    """Analyze potential outliers in the dataset"""
    model.eval()
    outlier_scores = []
    predictions = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            # Compute outlier scores directly
            scores = model.compute_outlier_score(images)
            # Get predictions
            logits = model(images)  # Model returns only logits in eval mode

            outlier_scores.extend(scores.cpu().numpy())
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy())

    outlier_scores = np.array(outlier_scores)
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Identify outliers using Z-score
    z_scores = (outlier_scores - np.mean(outlier_scores)) / np.std(outlier_scores)
    outliers = z_scores > threshold

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot 1: Outlier scores distribution
    plt.subplot(1, 2, 1)
    plt.hist(z_scores, bins=50)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.title('Distribution of Outlier Scores')
    plt.xlabel('Z-score')
    plt.ylabel('Count')
    plt.legend()

    # Plot 2: Scatter plot of outlier scores vs predictions
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(predictions, z_scores, c=labels, cmap='viridis', alpha=0.6)
    plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.title('Outlier Scores vs Predictions')
    plt.xlabel('Predicted Class')
    plt.ylabel('Outlier Score (Z-score)')
    plt.legend()
    plt.colorbar(scatter, label='True Class')

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\nFound {np.sum(outliers)} potential outliers out of {len(outlier_scores)} samples")
    print(f"Outlier percentage: {100 * np.sum(outliers) / len(outlier_scores):.2f}%")

    return outlier_scores, z_scores, outliers

def visualize_outliers(model, dataloader, device, threshold=3.0, num_samples=10):
    """Display sample images identified as outliers"""
    model.eval()
    classes = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']

    # Collect images and their outlier scores
    all_images = []
    all_scores = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # Get predictions
            logits = model(images)  # Model returns only logits in eval mode
            # Get outlier scores
            scores = model.compute_outlier_score(images)
            preds = torch.argmax(logits, dim=1)

            # Store batch data
            all_images.extend(images.cpu())
            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate z-scores
    z_scores = (all_scores - np.mean(all_scores)) / np.std(all_scores)

    # Find outlier indices
    outlier_indices = np.where(z_scores > threshold)[0]

    if len(outlier_indices) == 0:
        print("No outliers found with the current threshold.")
        return

    # Sort outliers by score for most extreme cases
    sorted_indices = outlier_indices[np.argsort(-z_scores[outlier_indices])]

    # Display top outliers
    n_cols = 5
    n_rows = (min(num_samples, len(sorted_indices)) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(20, 4*n_rows))

    for idx, outlier_idx in enumerate(sorted_indices[:num_samples]):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, xticks=[], yticks=[])

        # Get image and convert from tensor
        img = all_images[outlier_idx].numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)

        # Display image
        ax.imshow(img)

        # Add title with prediction and outlier score
        true_label = classes[all_labels[outlier_idx]]
        pred_label = classes[all_preds[outlier_idx]]
        score = z_scores[outlier_idx]

        title = f'True: {true_label}\nPred: {pred_label}\nOutlier Score: {score:.2f}'
        ax.set_title(title, color='red', fontsize=10)

    plt.suptitle(f'Top {num_samples} Outliers (Threshold = {threshold})', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nTotal outliers found: {len(outlier_indices)} out of {len(z_scores)} images")
    print(f"Percentage of outliers: {100 * len(outlier_indices) / len(z_scores):.2f}%")

    # Show class distribution of outliers
    print("\nClass distribution of outliers:")
    for i, cls in enumerate(classes):
        outlier_count = np.sum(all_labels[outlier_indices] == i)
        total_count = np.sum(all_labels == i)
        if total_count > 0:
            percentage = 100 * outlier_count / total_count
            print(f"{cls}: {outlier_count}/{total_count} ({percentage:.2f}%)")
