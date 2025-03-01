# Data paths
TRAIN_DATA_PATH = '/kaggle/input/UBC-OCEAN/train.csv'
TEST_DATA_PATH = '/kaggle/input/UBC-OCEAN/test.csv'
TRAIN_IMAGE_DIR = '/kaggle/input/UBC-OCEAN/train_thumbnails'
TEST_IMAGE_DIR = '/kaggle/input/UBC-OCEAN/test_thumbnails'

# Model parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
LEARNING_RATE = 5e-4
NUM_EPOCHS = 50
OUTLIER_EPOCHS = 30

# Data preprocessing parameters
WSI_MAGNIFICATION = 20.0
TMA_MAGNIFICATION = 40.0
STAIN_NORM_METHOD = 'reinhard'

# Class labels
CLASS_NAMES = ['HGSC', 'EC', 'CC', 'LGSC', 'MC']
LABEL_ENCODER = {'HGSC': 0, 'EC': 1, 'CC': 2, 'LGSC': 3, 'MC': 4}

# Model checkpoint paths
MODEL_CHECKPOINT_DIR = './model_checkpoints'
OUTLIER_CHECKPOINT_DIR = './outlier_model_checkpoints'
