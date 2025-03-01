import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import gc
import os
import albumentations as A

class OutlierTrainer:
    def __init__(self, model, train_loader, val_loader, device, criterion,
                 optimizer, scheduler, predictor, wsi_config, epochs=30,
                 save_dir='./outlier_model_checkpoints', batch_size=16):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.predictor = predictor
        self.wsi_config = wsi_config
        self.epochs = epochs
        self.save_dir = save_dir
        self.batch_size = batch_size

        # Create gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        os.makedirs(save_dir, exist_ok=True)

    def compute_auxiliary_loss(self, features, labels):
        """Auxiliary task: Feature clustering loss"""
        # Process in chunks to save memory
        chunk_size = 4
        total_center_loss = 0

        for i in range(0, len(features), chunk_size):
            chunk_features = features[i:i + chunk_size]
            chunk_labels = labels[i:i + chunk_size]

            # Compute center for each class in chunk
            centers = {}
            for cls in torch.unique(chunk_labels):
                centers[cls.item()] = chunk_features[chunk_labels == cls].mean(0)

            # Compute center loss for chunk
            chunk_loss = 0
            for cls in centers:
                cls_features = chunk_features[chunk_labels == cls]
                if len(cls_features) > 0:
                    chunk_loss += F.mse_loss(cls_features,
                                           centers[cls].expand(len(cls_features), -1))

            total_center_loss += chunk_loss

            # Clear cache
            torch.cuda.empty_cache()

        return total_center_loss

    def compute_consistency_loss(self, image, augmented_image):
        """Consistency between different views of same image"""
        # Process in chunks
        chunk_size = 4
        total_consist_loss = 0

        for i in range(0, len(image), chunk_size):
            # Extract features for original and augmented chunks
            with torch.cuda.amp.autocast():
                orig_features = self.model.extract_features(image[i:i + chunk_size])
                aug_features = self.model.extract_features(augmented_image[i:i + chunk_size])
                chunk_loss = F.mse_loss(orig_features, aug_features)

            total_consist_loss += chunk_loss

            # Clear cache
            torch.cuda.empty_cache()

        return total_consist_loss

    def augment_batch(self, images):
        """Memory efficient augmentation"""
        augmented = []
        chunk_size = 4

        for i in range(0, len(images), chunk_size):
            chunk = images[i:i + chunk_size]
            chunk_aug = []

            for img in chunk:
                transform = A.Compose([
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5)
                ])

                # Move to CPU for transformation
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                aug_img = transform(image=img_np)['image']
                chunk_aug.append(torch.from_numpy(aug_img.transpose(2, 0, 1)))

            # Move augmented chunk back to GPU
            chunk_tensor = torch.stack(chunk_aug).to(self.device)
            augmented.append(chunk_tensor)

            # Clear cache
            torch.cuda.empty_cache()

        return torch.cat(augmented, dim=0)

    def train_epoch(self):
        self.model.train()
        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_kl_loss = 0.0
        running_aux_loss = 0.0
        running_consist_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            # Limit batch size
            if len(inputs) > self.batch_size:
                inputs = inputs[:self.batch_size]
                labels = labels[:self.batch_size]

            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Clear cache before augmentation
            torch.cuda.empty_cache()

            # Get augmented version
            augmented_inputs = self.augment_batch(inputs)

            self.optimizer.zero_grad(set_to_none=True)

            # Use mixed precision training
            with torch.cuda.amp.autocast():
                # Get model outputs
                logits, mean, log_var = self.model(inputs)
                features = self.model.extract_features(inputs)

                # Calculate losses
                total_loss, ce_loss, kl_loss = self.criterion(logits, mean, log_var, labels)
                aux_loss = self.compute_auxiliary_loss(features, labels)
                consist_loss = self.compute_consistency_loss(inputs, augmented_inputs)

                # Combined loss
                total_loss = total_loss + 0.1 * aux_loss + 0.1 * consist_loss

            # Scaled backward pass
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            running_total_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_kl_loss += kl_loss.item()
            running_aux_loss += aux_loss.item()
            running_consist_loss += consist_loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Clear cache
            torch.cuda.empty_cache()

            pbar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        return {
            'total_loss': running_total_loss / len(self.train_loader),
            'ce_loss': running_ce_loss / len(self.train_loader),
            'kl_loss': running_kl_loss / len(self.train_loader),
            'aux_loss': running_aux_loss / len(self.train_loader),
            'consist_loss': running_consist_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_kl_loss = 0.0
        correct = 0
        total = 0

        # For outlier detection metrics
        all_scores = []
        all_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc='Validating')
        for inputs, labels in pbar:
            # Limit batch size
            if len(inputs) > self.batch_size:
                inputs = inputs[:self.batch_size]
                labels = labels[:self.batch_size]

            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Use mixed precision for validation too
            with torch.cuda.amp.autocast():
                # Get model outputs (model returns tuple in training mode)
                self.model.train()  # Temporarily set to train mode to get all outputs
                logits, mean, log_var = self.model(inputs)
                self.model.eval()  # Set back to eval mode

                # Calculate losses
                total_loss, ce_loss, kl_loss = self.criterion(logits, mean, log_var, labels)

                # Calculate outlier scores
                outlier_scores = self.model.compute_outlier_score(inputs)

            # Update metrics
            running_total_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_kl_loss += kl_loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store predictions and scores
            all_scores.extend(outlier_scores.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Clear cache
            torch.cuda.empty_cache()

            pbar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        # Calculate average metrics
        avg_total_loss = running_total_loss / len(self.val_loader)
        avg_ce_loss = running_ce_loss / len(self.val_loader)
        avg_kl_loss = running_kl_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        outlier_metrics = {
            'scores': np.array(all_scores),
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }

        return avg_total_loss, avg_ce_loss, avg_kl_loss, accuracy, outlier_metrics

    def train(self):
        print(f"\nStarting outlier detection training for {self.epochs} epochs...")
        best_metrics = {
            'best_val_loss': float('inf'),
            'best_val_acc': 0.0,
            'best_epoch': 0
        }

        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch+1}/{self.epochs}')
            print('-' * 20)

            # Training phase
            train_metrics = self.train_epoch()

            # Clear cache before validation
            torch.cuda.empty_cache()

            # Validation phase
            val_total_loss, val_ce_loss, val_kl_loss, val_acc, outlier_metrics = self.validate()

            # Print epoch results
            print(f'\nTraining Results:')
            print(f"Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"CE Loss: {train_metrics['ce_loss']:.4f}")
            print(f"KL Loss: {train_metrics['kl_loss']:.4f}")
            print(f"Auxiliary Loss: {train_metrics['aux_loss']:.4f}")
            print(f"Consistency Loss: {train_metrics['consist_loss']:.4f}")
            print(f"Accuracy: {train_metrics['accuracy']:.2f}%")

            print(f'\nValidation Results:')
            print(f'Total Loss: {val_total_loss:.4f}')
            print(f'CE Loss: {val_ce_loss:.4f}')
            print(f'KL Loss: {val_kl_loss:.4f}')
            print(f'Accuracy: {val_acc:.2f}%')

            # Save best model
            if val_total_loss < best_metrics['best_val_loss']:
                best_metrics['best_val_loss'] = val_total_loss
                best_metrics['best_val_acc'] = val_acc
                best_metrics['best_epoch'] = epoch + 1

                model_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'val_loss': val_total_loss,
                    'val_acc': val_acc,
                    'outlier_metrics': outlier_metrics
                }, model_path)
                print(f'Saved new best model with validation loss: {val_total_loss:.4f}')
                print(f'Saved new best model with validation accuracy: {val_acc:.2f}')

            # Memory cleanup after each epoch
            torch.cuda.empty_cache()
            gc.collect()

        return best_metrics
