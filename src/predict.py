import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from torch.nn import functional as F
import os
import pandas as pd
from config import *
from data_preprocessing import EnhancedPreprocessor
from utils import save_prediction_to_csv


def predict_single_image(image_path, model_path='./model_checkpoints/best_model.pth', device='cuda'):
    """
    Predict class for a single test image using the saved best model
    """
    try:
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize preprocessor
        preprocessor = EnhancedPreprocessor()
        preprocessed_image = preprocessor.preprocess_image(image)

        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image_tensor = transform(preprocessed_image).unsqueeze(0)

        # Load the saved model
        model = HistoPathModel()  # Initialize model architecture
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])  # Load weights
        model = model.to(device)
        model.eval()

        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        # Get prediction
        classes = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']
        pred_class = classes[torch.argmax(probabilities).item()]
        confidence = torch.max(probabilities).item()

        # Get probabilities for all classes
        class_probs = {cls: prob.item() for cls, prob in zip(classes, probabilities[0])}

        return pred_class, confidence, class_probs

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

def predict_batch(image_paths, model, device):
    """Predict classes for a batch of images"""
    predictions = []
    confidences = []
    all_probs = []

    preprocessor = EnhancedPreprocessor()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model.eval()
    with torch.no_grad():
        for image_path in image_paths:
            # Load and preprocess image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preprocessed_image = preprocessor.preprocess_image(image)
            image_tensor = transform(preprocessed_image).unsqueeze(0)

            # Make prediction
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Store results
            pred_class = CLASS_NAMES[torch.argmax(probabilities).item()]
            confidence = torch.max(probabilities).item()
            predictions.append(pred_class)
            confidences.append(confidence)
            all_probs.append(probabilities[0].cpu().numpy())

    return predictions, confidences, np.array(all_probs)

def create_submission(test_df, predictions, output_path='submission.csv'):
    """Create submission file with predictions"""
    submission_df = pd.DataFrame({
        'image_id': test_df['image_id'],
        'label': predictions
    })
    submission_df.to_csv(output_path, index=False)
    return submission_df

def main():
    """Main prediction pipeline"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"\nTest data shape: {test_df.shape}")
    print(test_df[['image_id', 'image_width', 'image_height']].head())

    try:
        # Make prediction for single test image
        test_image_path = os.path.join(TEST_IMAGE_DIR, f"{test_df['image_id'].iloc[0]}_thumbnail.png")
        print(f"\nMaking prediction for image: {test_df['image_id'].iloc[0]}")

        pred_class, confidence, class_probs = predict_single_image(
            test_image_path,
            MODEL_CHECKPOINT_DIR + '/best_model.pth',
            device
        )

        # Print detailed results
        print(f"\nPredicted class: {pred_class}")
        print(f"Confidence: {confidence:.2%}")
        print("\nClass probabilities:")
        for cls, prob in class_probs.items():
            print(f"{cls}: {prob:.2%}")

        # Save to CSV
        save_prediction_to_csv(test_df['image_id'].iloc[0], pred_class)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
