import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, balanced_accuracy_score
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import gc
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class MetricTracker:
    """Track training metrics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            'loss': [],
            'acc': [],
            'balanced_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_balanced_acc': [],
            'best_val_acc': 0.0,
            'best_val_balanced_acc': 0.0
        }

    def update(self, phase: str, loss: float, acc: float, balanced_acc: float):
        if phase == 'train':
            self.metrics['loss'].append(loss)
            self.metrics['acc'].append(acc)
            self.metrics['balanced_acc'].append(balanced_acc)
        else:
            self.metrics['val_loss'].append(loss)
            self.metrics['val_acc'].append(acc)
            self.metrics['val_balanced_acc'].append(balanced_acc)
            if acc > self.metrics['best_val_acc']:
                self.metrics['best_val_acc'] = acc
            if balanced_acc > self.metrics['best_val_balanced_acc']:
                self.metrics['best_val_balanced_acc'] = balanced_acc

    def get_best_metrics(self) -> Dict:
        return {
            'best_val_acc': self.metrics['best_val_acc'],
            'best_val_balanced_acc': self.metrics['best_val_balanced_acc'],
            'best_epoch_val_loss': min(self.metrics['val_loss']) if self.metrics['val_loss'] else float('inf'),
            'best_epoch_val_acc': max(self.metrics['val_acc']) if self.metrics['val_acc'] else 0,
            'best_epoch_val_balanced_acc': max(self.metrics['val_balanced_acc']) if self.metrics['val_balanced_acc'] else 0
        }

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, criterion,
                 optimizer, scheduler, predictor, wsi_config, epochs=50,
                 save_dir='./model_checkpoints'):
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
        self.tracker = MetricTracker()

        os.makedirs(save_dir, exist_ok=True)

    def compute_loss(self, main_logits, aux_logits, labels):
        """Compute combined loss from main and auxiliary outputs"""
        # Main classification loss
        main_loss = self.criterion(main_logits, labels)

        # Auxiliary classification loss
        aux_loss = self.criterion(aux_logits, labels)

        # Combined loss with weighting
        total_loss = main_loss + 0.3 * aux_loss

        return total_loss, main_loss, aux_loss

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Get both main and auxiliary outputs
            main_logits, aux_logits = self.model(inputs)

            # Compute combined loss
            loss, main_loss, aux_loss = self.compute_loss(main_logits, aux_logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item()
            _, predicted = main_logits.max(1)  # Use main classifier predictions
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        balanced_acc = 100. * balanced_accuracy_score(all_labels, all_preds)
        return running_loss / len(self.train_loader), 100. * correct / total, balanced_acc

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc='Validating')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                # During validation, model only returns main classifier output
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        balanced_acc = 100. * balanced_accuracy_score(all_labels, all_preds)

        class_names = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=3,
            output_dict=True
        )

        return running_loss / len(self.val_loader), 100. * correct / total, balanced_acc, report

    def train(self) -> Dict:
        print(f"\nStarting training for {self.epochs} epochs...")
        best_val_balanced_acc = 0.0

        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch+1}/{self.epochs}')
            print('-' * 20)

            # Get training metrics (now includes main and auxiliary losses)
            train_loss, train_acc, train_balanced_acc = self.train_epoch()
            self.tracker.update('train', train_loss, train_acc, train_balanced_acc)

            # Validation phase remains the same
            val_loss, val_acc, val_balanced_acc, report = self.validate()
            self.tracker.update('val', val_loss, val_acc, val_balanced_acc)

            # Print detailed training metrics
            print(f'\nTraining Results:')
            print(f'Total Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Balanced Acc: {train_balanced_acc:.2f}%')

            # Print validation metrics
            print(f'\nValidation Results:')
            print(f'Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Balanced Acc: {val_balanced_acc:.2f}%')

            # Print class-wise performance
            print('\nClass-wise Performance:')
            for cls_name in ['HGSC', 'EC', 'CC', 'LGSC', 'MC']:
                cls_metrics = report[cls_name]
                print(f'{cls_name} - Precision: {cls_metrics["precision"]:.3f}, '
                      f'Recall: {cls_metrics["recall"]:.3f}, '
                      f'F1: {cls_metrics["f1-score"]:.3f}')
            if val_balanced_acc > best_val_balanced_acc:
                best_val_balanced_acc = val_balanced_acc
                model_path = os.path.join(self.save_dir, 'best_model.pth')

                # Save model with additional metrics
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_balanced_acc': val_balanced_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_balanced_acc': train_balanced_acc,
                    'train_loss': train_loss,
                 }, model_path)
                print(f'Saved new best model with validation balanced accuracy: {val_balanced_acc:.2f}%')

            # Memory cleanup after each epoch
            torch.cuda.empty_cache()
            gc.collect()

        # Return final metrics
        best_metrics = self.tracker.get_best_metrics()

        print("\nTraining completed!")
        print("Best metrics achieved:")
        print(f"Best validation accuracy: {best_metrics['best_val_acc']:.2f}%")
        print(f"Best validation balanced accuracy: {best_metrics['best_val_balanced_acc']:.2f}%")
        print(f"Best epoch validation loss: {best_metrics['best_epoch_val_loss']:.4f}")

        return best_metrics
