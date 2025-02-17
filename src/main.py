import torch
import gc
from config import *
from data_preprocessing import prepare_data
from model import HistoPathModel
from trainer import Trainer
from outlier_model import OutlierHistoPathModel
from outlier_trainer import OutlierTrainer
from model_analysis import analyze_model
from outlier_analysis import analyze_outliers
from predict import predict_single_image

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Data preparation
        train_loader, val_loader = prepare_data(TRAIN_DATA_PATH, TRAIN_IMAGE_DIR, BATCH_SIZE)

        # Train standard model
        model = HistoPathModel(num_classes=NUM_CLASSES)
        trainer = Trainer(model, train_loader, val_loader, device)
        trainer.train()

        # Train outlier detection model
        outlier_model = OutlierHistoPathModel(num_classes=NUM_CLASSES)
        outlier_trainer = OutlierTrainer(outlier_model, train_loader, val_loader, device)
        outlier_trainer.train()

        # Analyze models
        analyze_model(model, train_loader, val_loader, device)
        analyze_outliers(outlier_model, val_loader, device)

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
