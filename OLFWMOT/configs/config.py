import os
from yacs.config import CfgNode as CN

__C = CN()

base_config = __C

__C.NAMES = CN()
__C.NAMES.NAME = "MOT16"

__C.PATHS = CN()
__C.PATHS.DATASET = "D:\BanYanDeng\MOTDataset"
__C.PATHS.MODEL_SAVE_DIR = "../output/models/MOT16"
__C.PATHS.RESULT_ROOT = "../output/results/MOT16"
__C.PATHS.EVAL_ROOT = "../output/evaluation/MOT16"

# Warm up
__C.WARMUP = CN()
__C.WARMUP.SUBNET = ['adaptive_head', 'memory_head']
__C.WARMUP.EPOCHS = 3
__C.WARMUP.MAX_LR = 0.00258
__C.WARMUP.LR_GAMMA = 0.5
__C.WARMUP.STEP_SIZE = 1
__C.WARMUP.OPTIM_NAME = "Adam"  # "SGD" or "Adam"
__C.WARMUP.LR_SCHEDULER = 'MultiStep'  # 'OneCycle' or 'MultiStep', 'StepLR
__C.WARMUP.LR_MILESTONES = [2, 4, 8]  # 'MultiStep' LR scheduler Milestone
__C.WARMUP.GRAD_CLIP = 3.
__C.WARMUP.DIV_FACTOR = 5    # 'OneCycle' LR scheduler parameter
__C.WARMUP.FINAL_DIV_FACTOR = 10     # 'OneCycle' LR scheduler parameter
__C.WARMUP.PCT_START = 0.2
__C.WARMUP.MOMENTUM = 0.937
__C.WARMUP.WEIGHT_DECAY = 0.000058

__C.TRAINING = CN()
# data setting
__C.TRAINING.BATCH_SIZE = 4
__C.TRAINING.SPLIT_BATCH_SIZE = 1
__C.TRAINING.SPLIT_LEN = 5
__C.TRAINING.WORKERS = 0

# Model
__C.TRAINING.BACKBONE_NAME = 'dla'  # default, resnet18, resnet34, resnet50, dla
__C.TRAINING.BOX_NUM = 3    # detect boxes number of every point
__C.TRAINING.BRIDGE_KERNEL = []     # conv kernel which make a balance among different experiment

# Choose subnet of the model to train, please refer to nets.StrongModel.StrongModel
# to learn more about the subnets available in the model
# "all" means to choose all subnet
__C.TRAINING.SUBNET = ['backbone', 'detect_head', 'level_split',
                       'adaptive_head', 'memory_head'
                       ]


# super parameters
__C.TRAINING.EPOCHS = 40
__C.TRAINING.MAX_LR = 0.001
__C.TRAINING.LR_GAMMA = 0.75
__C.TRAINING.STEP_SIZE = 2
__C.TRAINING.OPTIM_NAME = "SGD"  # "SGD" or "Adam"
__C.TRAINING.LR_SCHEDULER = 'StepLR'  # 'OneCycle' or 'MultiStep', 'StepLR'
__C.TRAINING.LR_MILESTONES = [2, 4, 8, 16]  # 'MultiStep' LR scheduler Milestone
__C.TRAINING.GRAD_CLIP = 3.
__C.TRAINING.DIV_FACTOR = 100    # 'OneCycle' LR scheduler parameter
__C.TRAINING.FINAL_DIV_FACTOR = 100     # 'OneCycle' LR scheduler parameter
__C.TRAINING.PCT_START = 0.3
__C.TRAINING.MOMENTUM = 0.937
__C.TRAINING.WEIGHT_DECAY = 0.00058
__C.TRAINING.VIS_THRESHOLD = 0.1

# pre-transform
__C.TRAINING.IMG_MEAN = [0., 0., 0.]
__C.TRAINING.IMG_STD = [1., 1., 1.]
__C.TRAINING.IMG_MIN_SIZE = [544, 608, 672, 704]
__C.TRAINING.IMG_MAX_SIZE = 1248
__C.TRAINING.MIN_AREA = 256.    # filters target less than 16. * 16. in size
__C.TRAINING.LEVEL_NUM = 2
__C.TRAINING.LEVEL_CHANNEL = 256
__C.TRAINING.SAVE_LEVELS = True
__C.TRAINING.SUBNET_MODE = "DETECT_ONLY"
__C.TRAINING.SPLIT_LEVEL_BY_VIS = False
__C.TRAINING.ROTATION_MAX_ANGLE = 12.
__C.TRAINING.ROTATION_MIN_SCALE = 0.75
__C.TRAINING.HGAIN = 0.02
__C.TRAINING.VGAIN = 0.74
__C.TRAINING.SGAIN = 0.34

# label setting
__C.TRAINING.BOX_LABEL_FUNC = 'default'     # default, patch
__C.TRAINING.LABEL_MIN_OVERLAP_RATIO = 0.7

# horizontal and vertical speed = ((cx, cy) - (cx_, cy_)) / (w, h)
# width and height speed = log((w_, h_) / (w, h))
# -SPEED_LIMIT < speed < SPEED_LIMIT
__C.TRAINING.SPEED_LIMIT = 4

# loss setting
__C.TRAINING.HEAT_MAP_LOSS = CN()
__C.TRAINING.HEAT_MAP_LOSS.LOSS_FUNC = "focal"  # 'focal', 'MSE'
__C.TRAINING.HEAT_MAP_LOSS.FOCAL_ALPHA = -1.   # -1: (pos_num / total_num, neg_num / total_num)
__C.TRAINING.HEAT_MAP_LOSS.BASE_WEIGHT = 1.
__C.TRAINING.HEAT_MAP_LOSS.FOCAL_BASE_ALPHA = 0.45

__C.TRAINING.BOX_LOSS = CN()
__C.TRAINING.BOX_LOSS.BASE_WEIGHT = 1.
__C.TRAINING.BOX_LOSS.EPS = 1e-5

__C.TRAINING.MEMORY_LOSS = CN()
__C.TRAINING.MEMORY_LOSS.FOCAL_ALPHA = -1.
__C.TRAINING.MEMORY_LOSS.BASE_WEIGHT = 1.
__C.TRAINING.MEMORY_LOSS.FOCAL_BASE_ALPHA = 0.2

# other
__C.TRAINING.PRINT_FREQ = 100
__C.TRAINING.SAVE_FREQ = 5
__C.TRAINING.RANDOM_SEED = 19990320
__C.TRAINING.VISUAL_FREQ = 100
__C.TRAINING.ENABLE_AUTOCAST = True
__C.TRAINING.SCALE_FACTOR = 16  # down sample rate of backbone
__C.TRAINING.CLAMP_EPS = 1e-8
__C.TRAINING.USE_SCALER = False
__C.TRAINING.GRADIENT_ACU_FRQ = 4       # gradient accumulate frequency

# tracking
__C.TRACKING = CN()
__C.TRACKING.MIN_BOX_SIZE = 3
__C.TRACKING.MIN_SCORE_ACTIVE_TRACKLET = 0.5
__C.TRACKING.MIN_SCORE_DETECTION = 0.1
__C.TRACKING.NMS_ACTIVE_TRACKLET = 0.7
__C.TRACKING.NMS_DETECTION = 0.3
__C.TRACKING.MIN_OVERLAP_AS_DISTRACTOR = 0.2
__C.TRACKING.MIN_RECOVER_GIOU = -0.4
__C.TRACKING.MIN_RECOVER_SCORE = 0.5
__C.TRACKING.MAX_LOST_FRAMES_BEFORE_REMOVE = 100


def load_config(config_file=None):
    """
        Load configurations
        Add or overwrite config from yaml file if specified
    """
    config = base_config
    if config_file is not None:
        config_file_path = os.path.join('../', "configs", f"{config_file}.yaml")
        # print('here', config_file_path)
        if os.path.isfile(config_file_path):
            config.merge_from_file(config_file_path)
            msg = f"Merged config from '{config_file_path}'"
        else:
            print(f"Cannot open the specified yaml config file '{config_file_path}'")
            exit(0)
    else:
        msg = f"No yaml config file is specified. Using default config."
    return config, msg
