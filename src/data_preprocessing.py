import numpy as np
import pandas as pd
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
from collections import Counter
import albumentations as A
from imblearn.over_sampling import SMOTE
from scipy.ndimage import gaussian_filter
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import torchvision.models as models
from tiatoolbox import logger
from tiatoolbox.tools import stainnorm, patchextraction
from tiatoolbox.tools.stainaugment import StainAugmentor
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import PIL.Image
import gc
import warnings
warnings.filterwarnings("ignore")

class EnhancedPreprocessor:
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 wsi_magnification: float = 20.0,
                 tma_magnification: float = 40.0,
                 stain_norm_method: str = 'reinhard'):
        self.target_size = target_size
        self.wsi_magnification = wsi_magnification
        self.tma_magnification = tma_magnification

        # Initialize stain normalizer
        if stain_norm_method == 'macenko':
            self.normalizer = stainnorm.MacenkoNormalizer()
        elif stain_norm_method == 'vahadane':
            self.normalizer = stainnorm.VahadaneNormalizer()
        elif stain_norm_method == 'reinhard':
            self.normalizer = stainnorm.ReinhardNormalizer()
        elif stain_norm_method == 'ruifrok':
            self.normalizer = stainnorm.RuifrokNormalizer()
        else:
            raise ValueError(f"Unknown stain normalization method: {stain_norm_method}")

    def detect_image_type(self, image: np.ndarray) -> str:
        """Determine if image is WSI or TMA based on size"""
        height, width = image.shape[:2]
        if height <= 5000 and width <= 5000:
            return 'TMA'
        return 'WSI'

    def normalize_magnification(self, image: np.ndarray, image_type: str) -> np.ndarray:
        """Normalize image magnification"""
        if image_type == 'TMA':
            scale_factor = self.wsi_magnification / self.tma_magnification
            new_size = (int(image.shape[1] * scale_factor),
                       int(image.shape[0] * scale_factor))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def apply_stain_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply stain normalization with error handling"""
        try:
            self.normalizer.fit(image)
            normalized = self.normalizer.transform(image)
            return normalized
        except Exception as e:
            print(f"Error in stain normalization: {str(e)}")
            return image  # Return original image if normalization fails

    def detect_tissue(self, image: np.ndarray) -> np.ndarray:
        """Improved tissue detection using LAB color space and adaptive thresholding"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        # Adaptive thresholding
        mask = cv2.adaptiveThreshold(
            l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask > 0

    def extract_tissue_region(self, image: np.ndarray) -> np.ndarray:
        """Extract main tissue region"""
        try:
            # Get tissue mask
            tissue_mask = self.detect_tissue(image)

            # Find contours
            contours, _ = cv2.findContours(tissue_mask.astype(np.uint8),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return image

            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Extract region with padding
            pad = 10
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(image.shape[1], x + w + pad)
            y_end = min(image.shape[0], y + h + pad)

            return image[y_start:y_end, x_start:x_end]

        except Exception as e:
            print(f"Error in tissue extraction: {str(e)}")
            return image

    def handle_image_dimensions(self, image: np.ndarray) -> np.ndarray:
        """Handle different image dimensions based on size"""
        height, width = image.shape[:2]

        # Handle very large WSIs
        if width > 50000 or height > 50000:
            scale_factor = min(50000 / width, 50000 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            print(f"Resizing large WSI from {width}x{height} to {new_width}x{new_height}")
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return image

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline"""
        try:
            # Handle large image dimensions first
            image = self.handle_image_dimensions(image)

            # Determine image type
            image_type = self.detect_image_type(image)

            # Normalize magnification
            image = self.normalize_magnification(image, image_type)

            # Extract tissue region
            image = self.extract_tissue_region(image)

            # Apply stain normalization
            image = self.apply_stain_normalization(image)

            # Resize to target size
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

            return image

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            # Return resized original image as fallback
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

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

class HistoDataset(Dataset):
    """Enhanced dataset with robust preprocessing"""
    def __init__(self, df: pd.DataFrame, image_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 is_training: bool = True,
                 apply_smote: bool = True,
                 stain_norm_method: str = 'reinhard'):

        self.df = df.copy()  # Make a copy to prevent modifications
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training
        self.apply_smote = apply_smote and is_training

        # Verify required columns exist
        required_columns = ['image_id', 'encoded_label']
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Initialize enhanced preprocessor
        self.preprocessor = EnhancedPreprocessor(
            stain_norm_method=stain_norm_method
        )

        # Load and preprocess images
        print("Loading and preprocessing images...")
        self._load_images()

        if self.apply_smote and len(self.processed_images) > 0:
            self._apply_smote_preprocessing()

    def _load_images(self):
        """Load and preprocess all images"""
        valid_indices = []
        self.processed_images = []
        self.labels = []

        for idx in tqdm(range(len(self.df)), desc="Processing images"):
            try:
                image_path = os.path.join(self.image_dir,
                                        f"{self.df.iloc[idx]['image_id']}_thumbnail.png")
                if os.path.exists(image_path):
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not read image - {image_path}")
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Preprocess image
                    processed_image = self.preprocessor.preprocess_image(image)

                    self.processed_images.append(processed_image)
                    valid_indices.append(idx)
                    self.labels.append(self.df.iloc[idx]['encoded_label'])
                else:
                    print(f"Warning: Image not found - {image_path}")

            except Exception as e:
                print(f"Error processing image at index {idx}: {str(e)}")
                continue

        if len(valid_indices) == 0:
            raise ValueError("No valid images were loaded")

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        self.labels = np.array(self.labels)

        print(f"Successfully processed {len(self.processed_images)} images")

    def _apply_smote_preprocessing(self):
        """Apply Borderline-SMOTE and random undersampling"""
        print("\nApplying SMOTE and undersampling...")
        print("Class distribution before resampling:")
        print(self.df['label'].value_counts())

        # Reshape images for SMOTE
        features = [img.reshape(-1) for img in self.processed_images]
        features = np.array(features)

        # Define resampling pipeline
        smote = BorderlineSMOTE(random_state=42)
        under = RandomUnderSampler(random_state=42)
        pipeline = Pipeline([('smote', smote), ('under', under)])

        # Apply resampling
        features_resampled, labels_resampled = pipeline.fit_resample(features, self.labels)

        # Reconstruct images
        self.images = [feat.reshape(self.preprocessor.target_size[0],
                                  self.preprocessor.target_size[1], 3)
                      for feat in features_resampled]

        # Create new balanced dataframe
        new_data = []
        for idx, label in enumerate(labels_resampled):
            new_data.append({
                'image_id': f'synthetic_{idx}' if idx >= len(self.df) else self.df.iloc[idx]['image_id'],
                'label': self.df['label'].unique()[label],
                'encoded_label': label,
                'is_synthetic': idx >= len(self.df)
            })

        self.df = pd.DataFrame(new_data)
        print("\nClass distribution after resampling:")
        print(self.df['label'].value_counts())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            if hasattr(self, 'images'):  # If SMOTE was applied
                image = self.images[idx]
                label = self.df.iloc[idx]['encoded_label']
            else:  # Original image loading
                image = self.processed_images[idx]
                label = self.labels[idx]

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            # Convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            label = torch.tensor(label, dtype=torch.long)

            return image, label

        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            return torch.zeros((3, 224, 224)), torch.tensor(0)

def prepare_data(df: pd.DataFrame, image_dir: str, batch_size: int = 32):
    """Prepare data loaders with TIAToolbox preprocessing"""
    try:
        # Create label encodings
        label_encoder = {'HGSC': 0, 'EC': 1, 'CC': 2, 'LGSC': 3, 'MC': 4}
        df['encoded_label'] = df['label'].map(label_encoder)

        # Stratified split
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df['label'],
            random_state=42
        )

        # Create transforms
        train_transform, val_transform = create_transforms(image_size=(224, 224), stain_augment_prob=0.5 )

        print("Creating training dataset...")
        train_dataset = HistoDataset(
            df=train_df,
            image_dir=image_dir,
            transform=train_transform,
            is_training=True,
            apply_smote=True,
            stain_norm_method='reinhard',# try 'macenko' or 'reinhard' or 'ruifrok'
        )

        print("\nCreating validation dataset...")
        val_dataset = HistoDataset(
            df=val_df,
            image_dir=image_dir,
            transform=val_transform,
            is_training=False,
            apply_smote=False,
            stain_norm_method='reinhard',
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        return train_loader, val_loader

    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        raise

def main():
    """Main function to prepare and store data loaders in global scope"""
    try:
        # Load data
        df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
        image_dir = '/kaggle/input/UBC-OCEAN/train_thumbnails'

        print("Initial class distribution:")
        print(df['label'].value_counts())

        # Create data loaders with full pipeline and store in global scope
        global train_loader, val_loader
        train_loader, val_loader = prepare_data(df, image_dir)

        # Test batch loading
        images, labels = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"Images: {images.shape}")
        print(f"Labels: {labels.shape}")

        print("\nClass distribution in batch:")
        print(pd.Series(labels.numpy()).value_counts())

        # Show sample images
        print("\nDisplaying sample processed images:")
        show_sample_images(train_loader)

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in main: {str(e)}")
        # Memory cleanup even if there's an error
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
