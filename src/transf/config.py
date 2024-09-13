
import os

# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


VALIDATION_SPLIT = 0.20
SEED = 1234
N_SPLITS = 5
N_REPEATS = 2