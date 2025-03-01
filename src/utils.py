import torch
import numpy as np
import pandas pd
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from sklearn.metrics import classification_report, balanced_accuracy_score

def create_transforms(image_size: Tuple[int, int] = (224, 224), stain_augment_prob: float = 0.5):
    """Create augmentation transforms with advanced techniques"""
    train_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1], always_apply=True),
        # Color augmentations
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.RandomGamma(p=1.0)
        ], p=0.5),
        # Geometric augmentations
        A.OneOf([
            A.ElasticTransform(p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5)
        ], p=0.3),
        # Cutout augmentation
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
        # Basic transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ])

    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1], always_apply=True)
    ])

    return train_transform, val_transform

def visualize_sample_predictions(model, val_loader, device, num_samples=10):
    """Display sample images with their predictions"""
    model.eval()
    classes = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']

    # Get a batch of images
    images, labels = next(iter(val_loader))

    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        if isinstance(outputs, tuple):  # Handle training mode output
            outputs = outputs[0]
        _, preds = torch.max(outputs, 1)

    # Create a figure to display images
    fig = plt.figure(figsize=(20, 4))
    for idx in range(min(num_samples, len(images))):
        ax = fig.add_subplot(1, num_samples, idx + 1, xticks=[], yticks=[])

        # Convert tensor to image
        img = images[idx].numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)

        # Display image
        ax.imshow(img)

        # Add title with true and predicted labels
        true_label = classes[labels[idx]]
        pred_label = classes[preds[idx].cpu()]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)

    plt.tight_layout()
    plt.show()

def plot_feature_distribution(features, labels):
    """Plot the distribution of features across classes"""
    # Reduce dimensionality to 2D using UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding = reducer.fit_transform(features)

    # Create scatter plot with different colors for each class
    plt.figure(figsize=(12, 8))
    classes = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, cls in enumerate(classes):
        mask = labels == i
        plt.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=colors[i], label=cls, alpha=0.6)

    plt.title('Feature Distribution Across Classes')
    plt.legend()
    plt.show()

def plot_feature_space(features, labels, title="Feature Space Visualization"):
    """Create UMAP visualization of feature space"""
    # Reduce dimensionality to 2D
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding = reducer.fit_transform(features)

    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')
    plt.colorbar(scatter, label='True Class')
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()

def save_prediction_to_csv(image_id, predicted_class, output_path='submission.csv'):
    """
    Save prediction to CSV in the required format
    """
    df = pd.DataFrame({
        'image_id': [image_id],
        'label': [predicted_class]
    })
    df.to_csv(output_path, index=False)
    print(f"\nPrediction saved to {output_path}")
    print(f"Content preview:")
    print(df.to_string(index=False))
