import os
import torch
import logging
import argparse
from models import build_model
from Transformer.Transformer import MetadataOnlyClassifier as TransformerMetadataOnlyClassifier
from Transformer.Transformer import IntermediateFusionClassifier as TransformerIntermediateFusionClassifier
from Transformer.Transformer import LateFusionClassifier as TransformerLateFusionClassifier
from CNN.CNN import MetadataOnlyClassifier as CNNMetadataOnlyClassifier
from CNN.CNN import IntermediateFusionClassifier as CNNIntermediateFusionClassifier
from CNN.CNN import LateFusionClassifier as CNNLateFusionClassifier

class argparse_namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def parse_option():
    parser = argparse.ArgumentParser('Unified Training Script', add_help=False)
    parser.add_argument('--model_to_run', type=str, required=True, choices=['cnn', 'transformer'], help='Model name to run')
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


def load_dotenv(dotenv_path):
    if not os.path.isfile(dotenv_path):
        return

    with open(dotenv_path, 'r', encoding='utf-8') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def apply_cli_overrides(config, args):
    config.EXPERIMENT_NAME = args.experiment_name or config.EXPERIMENT_NAME
    config.DATA.INPUT_MODE = args.input_mode
    if args.data_path is not None:
        config.DATA.DATA_PATH = args.data_path
    if args.images_dir is not None:
        config.DATA.IMAGES_DIR = args.images_dir
    if args.metadata_path is not None:
        config.DATA.METADATA_PATH = args.metadata_path
    if args.cleaned_metadata_path is not None:
        config.DATA.CLEANED_METADATA_PATH = args.cleaned_metadata_path
    if args.image_column is not None:
        config.DATA.IMAGE_COLUMN = args.image_column
    if args.label_column is not None:
        config.DATA.LABEL_COLUMN = args.label_column
    if args.split_column is not None:
        config.DATA.SPLIT_COLUMN = args.split_column
    if args.val_split is not None:
        config.DATA.VAL_SPLIT = args.val_split
    if args.return_metadata or args.input_mode in {'metadata_only', 'intermediate_fusion', 'late_fusion'}:
        config.DATA.RETURN_METADATA = True

    if args.wandb_project is not None:
        config.WANDB.PROJECT = args.wandb_project
    if args.wandb_entity is not None:
        config.WANDB.ENTITY = args.wandb_entity
    if args.disable_wandb:
        config.WANDB.ENABLE = False


def extract_batch(batch):
    if isinstance(batch, tuple):
        samples, labels = batch
        return {'image': samples, 'label': labels}
    if isinstance(batch, dict):
        return batch
    raise TypeError(f'Unsupported batch format: {type(batch)}')


def _resolve_fusion_classes(model_name):
    lowered = model_name.lower()
    if lowered == 'cnn':
        return (
            CNNMetadataOnlyClassifier,
            CNNIntermediateFusionClassifier,
            CNNLateFusionClassifier,
        )
    return (
        TransformerMetadataOnlyClassifier,
        TransformerIntermediateFusionClassifier,
        TransformerLateFusionClassifier,
    )


def create_training_model(config, metadata_dim):
    input_mode = config.DATA.INPUT_MODE
    num_classes = config.MODEL.NUM_CLASSES

    if input_mode == 'images_only':
        return build_model(config)

    metadata_cls, intermediate_cls, late_cls = _resolve_fusion_classes(config.MODEL.NAME)

    if input_mode == 'metadata_only':
        return metadata_cls(metadata_dim=metadata_dim, num_classes=num_classes)

    image_model = build_model(config)
    if input_mode == 'intermediate_fusion':
        return intermediate_cls(image_model=image_model, metadata_dim=metadata_dim, num_classes=num_classes)
    if input_mode == 'late_fusion':
        return late_cls(image_model=image_model, metadata_dim=metadata_dim, num_classes=num_classes)

    raise ValueError(f'Unsupported input mode: {input_mode}')


def forward_by_mode(model, batch, input_mode):
    target = batch['label'].cuda(non_blocking=True)

    if input_mode == 'images_only':
        image = batch['image'].cuda(non_blocking=True)
        output = model(image)
        return output, target
    if input_mode == 'metadata_only':
        metadata_tensor = batch['metadata_tensor'].cuda(non_blocking=True)
        output = model(metadata_tensor)
        return output, target
    if input_mode in {'intermediate_fusion', 'late_fusion'}:
        image = batch['image'].cuda(non_blocking=True)
        metadata_tensor = batch['metadata_tensor'].cuda(non_blocking=True)
        output = model(image, metadata_tensor)
        return output, target

    raise ValueError(f'Unsupported input mode: {input_mode}')


def init_wandb(config, logger):
    if not config.WANDB.ENABLE:
        logger.info('W&B logging disabled')
        return None

    try:
        import wandb
    except ImportError:
        logger.info('wandb not installed; skipping W&B logging')
        return None

    run_name = config.EXPERIMENT_NAME or config.WANDB.RUN_NAME or config.MODEL.NAME
    run = wandb.init(
        project=config.WANDB.PROJECT,
        entity=config.WANDB.ENTITY,
        name=run_name,
        config={
            'model_name': config.MODEL.NAME,
            'data_path': config.DATA.DATA_PATH,
            'images_dir': config.DATA.IMAGES_DIR,
            'metadata_path': config.DATA.METADATA_PATH,
            'cleaned_metadata_path': config.DATA.CLEANED_METADATA_PATH,
            'batch_size': config.DATA.BATCH_SIZE,
            'epochs': config.TRAIN.EPOCHS,
            'input_mode': config.DATA.INPUT_MODE,
        },
    )
    logger.info(f'W&B run initialized: {run.name}')
    return run
