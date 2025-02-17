import torch
import gc
from config import *
from model import HistoPathModel, create_model
from trainer import Trainer
from data_preprocessing import prepare_data

def train_main():
    try:
        print("Initializing Training Pipeline...")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Prepare data
        train_loader, val_loader = prepare_data(
            TRAIN_DATA_PATH,
            TRAIN_IMAGE_DIR,
            BATCH_SIZE
        )

        print("\nInitializing model...")
        # Create model and get all components
        model, criterion, optimizer, scheduler, predictor, wsi_config = create_model(
            device,
            learning_rate=LEARNING_RATE,
        )

        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            predictor=predictor,
            wsi_config=wsi_config,
            epochs=NUM_EPOCHS,
            save_dir=MODEL_CHECKPOINT_DIR
        )

        # Start training
        print("\nStarting training process...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")

        best_metrics = trainer.train()

        # Print final results
        print("\nTraining completed!")
        print("Best metrics achieved:")
        print(f"Best validation accuracy: {best_metrics['best_val_acc']:.2f}%")
        print(f"Best validation balanced accuracy: {best_metrics['best_val_balanced_acc']:.2f}%")
        print(f"Best epoch validation loss: {best_metrics['best_epoch_val_loss']:.4f}")

        # Clean up
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

        return best_metrics

    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

        # Clean up even if there's an error
        try:
            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        return None

if __name__ == "__main__":
    train_main()
