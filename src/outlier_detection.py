import torch
import gc
from config import *
from outlier_model import OutlierHistoPathModel, OutlierLoss, create_outlier_model
from outlier_trainer import OutlierTrainer
from outlier_analysis import analyze_outliers, visualize_outliers

def train_outlier_main():
    try:
        print("Initializing Outlier Detection Pipeline...")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Use global data loaders
        global train_loader, val_loader

        print("\nInitializing model with outlier detection...")
        # Create model with outlier detection capabilities
        model, criterion, optimizer, scheduler, predictor, wsi_config = create_outlier_model(
            device,
            learning_rate=5e-4,
            epochs=30
        )

        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Initialize trainer with memory optimizations
        trainer = OutlierTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            predictor=predictor,
            wsi_config=wsi_config,
            epochs=30,
            save_dir='./outlier_model_checkpoints',
            batch_size=16  # Add reduced batch size for memory efficiency
        )

        # Start training
        print("\nStarting training process...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")

        best_metrics = trainer.train()

        # Print final results
        print("\nTraining completed!")
        print("Best metrics achieved:")
        print(f"Best validation loss: {best_metrics['best_val_loss']:.4f}")
        print(f"Best validation accuracy: {best_metrics['best_val_acc']:.2f}%")
        print(f"Best epoch: {best_metrics['best_epoch']}")

        # Clean up
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

        return best_metrics

    except Exception as e:
        print(f"Error in outlier detection pipeline: {str(e)}")
        traceback.print_exc()

        # Clean up even if there's an error
        try:
            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        return None

#if __name__ == "__main__":
    #train_outlier_main()

def run_outlier_analysis():
    """Run the complete outlier analysis pipeline"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load trained model
        model = OutlierHistoPathModel()
        model.load_state_dict(torch.load('./outlier_model_checkpoints/best_model.pth')['model_state_dict'])
        model.to(device)

        # Different thresholds for comparison
        thresholds = [2.0, 2.5, 3.0]
        results = {}

        for threshold in thresholds:
            print(f"\nAnalyzing outliers with threshold {threshold}...")
            outlier_scores, z_scores, outliers = analyze_outliers(
                model,
                val_loader,
                device,
                threshold=threshold
            )
            results[threshold] = {
                'scores': outlier_scores,
                'z_scores': z_scores,
                'outliers': outliers
            }

            # Visualize outliers for each threshold
            print(f"\nVisualizing outliers for threshold {threshold}...")
            visualize_outliers(model, val_loader, device, threshold=threshold)

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"Error in outlier analysis: {str(e)}")
        traceback.print_exc()
        return None

#run_outlier_analysis()

def main():
    try:
        print("Starting Outlier Detection Pipeline...")

        # Train outlier detection model
        best_metrics = train_outlier_main()

        if best_metrics:
            print("\nStarting Outlier Analysis...")
            results = run_outlier_analysis()

            # Print summary of findings
            print("\nOutlier Detection Summary:")
            for threshold, data in results.items():
                print(f"\nThreshold {threshold}:")
                print(f"Total outliers: {np.sum(data['outliers'])}")
                print(f"Outlier percentage: {100 * np.sum(data['outliers']) / len(data['outliers']):.2f}%")

    except Exception as e:
        print(f"Error in outlier detection pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
