import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# dataset
TRAIN_PATH = r"C:\Users\Hussein\Documents\Github\AI Projects\lab4_cv\dataset\train"
VAL_PATH   = r"C:\Users\Hussein\Documents\Github\AI Projects\lab4_cv\dataset\valid"
TEST_PATH  = r"C:\Users\Hussein\Documents\Github\AI Projects\lab4_cv\dataset\test"

NUM_CLASSES = 4
IMG_SIZE = 224
VAL_SPLIT = 0.2

# training
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoints
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

INCEPTION_IMG_SIZE = 299