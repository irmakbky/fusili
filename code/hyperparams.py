REGULARIZATION = False
REG_LAMBDA = 0.01
IMAGE_SIZE = 224
MAX_AUDIO_LEN = 20
TAU = 0.07
NUM_EPOCHS = 20
n_iters = 20
alpha = 0.5
EARLY_STOP = False
trim_audio = True
GRADIENT_CKPT = True
PAD_TO_SQUARE = False
DETACH = False
LR = 2e-4
LR_FACTOR = 0.1
TEMPERATURE = 0.07
WARMUP_EPOCHS = 1
SPLIT = "94/5/1" # default: "80/10/10"

MODEL_CKPT = None
START_EPOCH = 0

# PDMX = True
UNWRAPPED = False
NEG_BATCH = True
SEPARATE_BY_CHUNK = True
RANDOM_BATCH = True
NEG_BATCH_DIR = None # /path/to/negative/batch/data
TRAIN_CSV = None # /path/to/train/csv for mutations
METHOD = "sinkhorn_global" # local_sinkhorn, baseline, etc
BATCH_SIZE = 32

FREEZE_ENCODERS = False
UNFREEZE_EPOCH = 0
ENABLE_DDP = True 
LOG_WANDB = True
project_name = "pdmx"

CSV_PATH = "/path/to/train/data/csv" # TODO, train data csv
CLIP_CKPT_PATH = ["ViT-B-32", "laion2b_s34b_b79k"] # ["ViT-B-32", None]
CLAP_CKPT_PATH = ["HTSAT-base", '/path/to/clap_ckpts/music_audioset_epoch_15_esc_90.14.pt'] # ["HTSAT-base", None]
if "baseline" in METHOD:
    method_save_name = f"{METHOD}"
elif "sinkhorn" in METHOD or "softmax" in METHOD:
    method_save_name = f"{METHOD}_tau_{TAU}"
else:
    raise Exception()
CHECKPOINT_DIR = f"/path/to/checkpoint/dir/{method_save_name}/"
wandb_run_name = method_save_name
if ENABLE_DDP:
    wandb_run_name = "ddp_" + wandb_run_name
