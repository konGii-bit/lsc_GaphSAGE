import os

# ---------- Paths ----------
DATA_PATH = 'interactions_modified1.csv'
RESULTS_DIR = 'results'
TB_LOG_DIR = 'runs'
ROC_PATH = None

DEVICE = 'cuda'

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)

# ---------- Hyperparameters ----------
EMBED_DIM = 32
HIDDEN_DIM = 64
LEARNING_RATE = 0.01
EPOCHS = 1
NEG_SAMPLING_METHOD = 'sparse'  # can be 'dense' or 'sparse'
TEST_SPLIT = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 4
NUM_NEIGHBORS  = 8