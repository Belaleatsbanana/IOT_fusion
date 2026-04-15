import os
import copy

# Base Configuration with all possible hyperparameters
BASE_CONFIG = {
    'DATA': {
        'DATA_PATH': '/content/drive/MyDrive/iot/pad_ufes_20',
        'IMAGES_DIR': None,
        'METADATA_PATH': None,
        'CLEANED_METADATA_PATH': None,
        'IMAGE_COLUMN': None,
        'LABEL_COLUMN': None,
        'SPLIT_COLUMN': None,
        'VAL_SPLIT': 0.2,
        'SEED': 42,
        'RETURN_METADATA': False,
        'INPUT_MODE': 'images_only',
        'BATCH_SIZE': 64,
        'NUM_WORKERS': 2,
        'PIN_MEMORY': True,
    },
    'MODEL': {
        'NAME': None, # Must be specified by model config
        'TAG': None,  # Architecture type (CNN, MLP, Transformer)
        'NUM_CLASSES': 5,
        'DROP_PATH_RATE': 0.1,
        'LABEL_SMOOTHING': 0.1,
    },
    'TRAIN': {
        'START_EPOCH': 0,
        'EPOCHS': 100,
        'BASE_LR': 5e-4,
        'WEIGHT_DECAY': 0.05,
        'CLIP_GRAD': 5.0,
        'WARMUP_EPOCHS': 5,
        'WARMUP_LR': 1e-6,
        'MIN_LR': 1e-5,
        'OPT': 'adamw',
        'SCHED': 'cosine',
        'MOMENTUM': 0.9,
        'EPS': 1e-8,
        'BETAS': (0.9, 0.999),
        'COOLDOWN_EPOCHS': 0,
        'DECAY_EPOCHS': 30,
        'DECAY_RATE': 0.1,
    },
    'WANDB': {
        'ENABLE': True,
        'PROJECT': 'iot-pad-ufes-20',
        'ENTITY': None,
        'RUN_NAME': None,
    },
    'EXPERIMENT_NAME': None,
    'OUTPUT': 'outputs',
    'AMP_ENABLE': True,
}

class Config:
    def __init__(self, model_config):
        merged = copy.deepcopy(BASE_CONFIG)
        # Merge model_config into BASE_CONFIG
        for section, params in model_config.items():
            if isinstance(params, dict) and section in merged and isinstance(merged[section], dict):
                for k, v in params.items():
                    if v is not None:
                        merged[section][k] = v
            elif params is not None:
                merged[section] = params
        
        # Set attributes from merged config
        self.__dict__.update(merged)
        
        # Convert dicts to simple objects for dot notation
        self.DATA = type('', (), self.DATA)()
        self.MODEL = type('', (), self.MODEL)()
        self.TRAIN = type('', (), self.TRAIN)()
        self.WANDB = type('', (), self.WANDB)()
        
        # Ensure output dir exists
        os.makedirs(self.OUTPUT, exist_ok=True)

    def defrost(self): pass
    def freeze(self): pass
