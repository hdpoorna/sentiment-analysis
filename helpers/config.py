"""
py38
hdpoorna
"""

# constants
BATCH_SIZE = 32
VAL_SPLIT = 0.2
DATA_SUBSET = "both"  # None if VAL_SPLIT is None, else "training" or "both"
SEED = 42

MAX_FEATURES = 100000       # good enough for IMDb dataset according to its vocabulary data
MAX_SEQUENCE_LENGTH = 2500
SEQUENCE_LENGTH = None  # 250 or 2500 or MAX_SEQUENCE_LENGTH to limit and pad; None for unlimited

EMBEDDING_DIM = 32  # 16 or 32 or 64

# for transformer block
NUM_ATTN_HEADS = 2
FEED_FWD_DIM = EMBEDDING_DIM

MASK_ZERO = True  # True if SEQUENCE_LENGTH = None

EPOCHS = 10
LEARNING_RATE = 1e-3

AVG_MODEL_DROPOUTS = [{"apply": True, "rate": 0.2},
                      {"apply": True, "rate": 0.2},
                      {"apply": True, "rate": 0.5}]  # 3 dropouts

RNN_MODEL_DROPOUTS = [{"apply": True, "rate": 0.2},
                      {"apply": True, "rate": 0.2},
                      {"apply": True, "rate": 0.5}]  # 3 dropouts

TRANSFORMER_MODEL_DROPOUTS = [{"apply": True, "rate": 0.2},
                              {"apply": True, "rate": 0.1},
                              {"apply": True, "rate": 0.1}]  # 3 dropouts

MODEL_DROPOUTS = [{"apply": False, "rate": None},
                  {"apply": True, "rate": 0.2},
                  {"apply": True, "rate": 0.2}]  # 3 dropouts


