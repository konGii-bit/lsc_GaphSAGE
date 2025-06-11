import os

# ---------- Paths ----------
DATA_PATH = 'interactions_modified1.csv'
RESULTS_DIR = 'results'
TB_LOG_DIR = 'runs'
ROC_PATH = None
DEVICE = 'cuda'

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)

# ---------- Hyperparameters ----------
EMBED_DIM = 16
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
EPOCHS = 2
BATCH_SIZE = 1024 
NEG_SAMPLING_RATIO = 1 
TEST_SPLIT = 0.2
VALID_SPLIT = 0.1
RANDOM_SEED = 42