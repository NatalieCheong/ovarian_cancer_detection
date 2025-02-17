import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import torch
import gc

def extract_features(model, dataloader, device):
    """Extract features from the model's intermediate layer"""
    features = []
    positions = []
    labels = []

    model.eval()
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(dataloader):
            images = images.to(device)
            # Use model's extract_features method directly
            batch_features = model.extract_features(images)

            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())
            positions.append(np.array([(i * images.shape[0] + j, j) for j in range(images.shape[0])]))

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    positions = np.concatenate(positions)
    return features, labels, positions

def visualize_predictions(model, dataloader, device):
    """Visualize model predictions vs actual labels with enhanced metrics"""
    predictions = []
    actuals = []
    probabilities = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):  # Handle training mode output
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.numpy())
            probabilities.extend(probs.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    probabilities = np.array(probabilities)

    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(actuals, predictions) * 100

    # Create confusion matrix plot
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(12, 8))
    classes = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix\nBalanced Accuracy: {balanced_acc:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(actuals, predictions, target_names=classes))

    # Plot per-class prediction confidence
    plt.figure(figsize=(12, 6))

    for i, cls in enumerate(classes):
        true_mask = actuals == i
        pred_mask = predictions == i

        # Correct predictions
        correct_mask = np.logical_and(true_mask, pred_mask)
        if np.any(correct_mask):
            plt.scatter(np.full(np.sum(correct_mask), i+0.1),
                       probabilities[correct_mask, i],
                       c='green', alpha=0.5, label='Correct' if i == 0 else '')

        # Wrong predictions
        wrong_mask = np.logical_and(true_mask, ~pred_mask)
        if np.any(wrong_mask):
            plt.scatter(np.full(np.sum(wrong_mask), i-0.1),
                       probabilities[wrong_mask, i],
                       c='red', alpha=0.5, label='Wrong' if i == 0 else '')

    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.ylabel('Prediction Confidence')
    plt.title('Per-class Prediction Confidence Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return balanced_acc, cm, predictions, actuals, probabilities


def analyze_model(model, train_loader, val_loader, device):
    """Run comprehensive model analysis with enhanced visualizations"""
    print("1. Displaying sample predictions...")
    visualize_sample_predictions(model, val_loader, device)

    print("\n2. Extracting features...")
    train_features, train_labels, _ = extract_features(model, train_loader, device)
    val_features, val_labels, _ = extract_features(model, val_loader, device)

    print("\n3. Plotting feature distributions...")
    plot_feature_distribution(train_features, train_labels)

    print("\n4. Analyzing predictions and metrics...")
    balanced_acc, cm, predictions, actuals, probs = visualize_predictions(model, val_loader, device)

    print(f"\nOverall Balanced Accuracy: {balanced_acc:.2f}%")

    print("\n5. Plotting UMAP embeddings...")
    plot_feature_space(train_features, train_labels, "Training Data Feature Space")
    plot_feature_space(val_features, val_labels, "Validation Data Feature Space")

    # Additional per-class analysis
    print("\nPer-class Performance Summary:")
    for i, cls in enumerate(['HGSC', 'EC', 'CC', 'LGSC', 'MC']):
        class_mask = actuals == i
        class_acc = balanced_accuracy_score([1 if x == i else 0 for x in actuals],
                                          [1 if x == i else 0 for x in predictions]) * 100
        class_conf = probs[class_mask, i].mean() * 100
        print(f"{cls}:")
        print(f" Balanced Accuracy: {class_acc:.2f}%")
        print(f" Average Confidence: {class_conf:.2f}%")

def run_analysis():
    """Run the complete analysis pipeline"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load your best model
        model = HistoPathModel()
        checkpoint = torch.load('./model_checkpoints/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print("\nStarting model analysis...")
        print(f"Model checkpoint metrics:")
        print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"Validation Balanced Accuracy: {checkpoint['val_balanced_acc']:.2f}%")
        print(f"Validation Loss: {checkpoint['val_loss']:.4f}")

        analyze_model(model, train_loader, val_loader, device)

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    run_analysis()
