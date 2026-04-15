def get_config(): # resnext50_32x4d
    return {
        'MODEL': {
            'NAME': 'resnext50_local',
            'TAG': 'CNN',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.15,
            'LABEL_SMOOTHING': 0.1,
            'NUM_BLOCKS': [3, 4, 6, 3],
            'CARDINALITY': 32,
            'BOTTLENECK_WIDTH': 4,
            'EXPANSION': 2,
            'KERNEL_SIZE': 7,
            'STRIDE': 2,
            'PADDING': 3
        },
        'DATA': {
            'DATA_PATH': None,
            'IMAGES_DIR': None,
            'METADATA_PATH': None,
            'CLEANED_METADATA_PATH': None,
            'IMAGE_COLUMN': None,
            'LABEL_COLUMN': None,
            'SPLIT_COLUMN': None,
            'VAL_SPLIT': None,
            'RETURN_METADATA': None,
            'BATCH_SIZE': None,
            'NUM_WORKERS': None,
            'PIN_MEMORY': None
        },
        'TRAIN': {
            'START_EPOCH': None,
            'EPOCHS': 100,
            'BASE_LR': 5e-4,
            'WEIGHT_DECAY': 5e-4,
            'CLIP_GRAD': 1.0,
            'WARMUP_EPOCHS': 5,
            'WARMUP_LR': 1e-6,
            'MIN_LR': 1e-6,
            'OPT': 'adamw',
            'SCHED': 'cosine'
        },
        'OUTPUT': None
    }
