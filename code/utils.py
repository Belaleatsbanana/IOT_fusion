import os
import torch
import logging
import argparse
from timm.utils import AverageMeter

class argparse_namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def parse_option():
    parser = argparse.ArgumentParser('Unified Training Script', add_help=False)
    parser.add_argument('--model_to_run', type=str, required=True, help='Model name to run (e.g., as_mlp_tiny, deit_tiny, resnext50_local)')
    parser.add_argument('--input_mode', type=str, default='images_only', choices=['images_only', 'metadata_only', 'intermediate_fusion', 'late_fusion'], help='Training input mode')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name used for output files and W&B run name')
    parser.add_argument('--data_path', type=str, default=None, help='Dataset root path (expects images/ and metadata CSV files)')
    parser.add_argument('--images_dir', type=str, default=None, help='Path to image directory')
    parser.add_argument('--metadata_path', type=str, default=None, help='Path to metadata CSV')
    parser.add_argument('--cleaned_metadata_path', type=str, default=None, help='Path to cleaned metadata CSV (takes precedence over metadata_path)')
    parser.add_argument('--image_column', type=str, default=None, help='Metadata column containing image identifiers/paths')
    parser.add_argument('--label_column', type=str, default=None, help='Metadata column containing labels')
    parser.add_argument('--split_column', type=str, default=None, help='Metadata column containing split names')
    parser.add_argument('--val_split', type=float, default=None, help='Validation ratio when no split column is available')
    parser.add_argument('--return_metadata', action='store_true', help='Return metadata in each batch for fusion workflows')
    parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity/team')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases logging')
    args = parser.parse_args()
    return args

def create_logger(output_dir, name):
    # Setup logging to both console and file
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Create formatters
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, f'{name}_log.txt'), mode='a')
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    return logger
