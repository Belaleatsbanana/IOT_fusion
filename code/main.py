import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import csv
from pathlib import Path
from timm.utils import accuracy, AverageMeter
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Local imports
from models import build_model, get_model_config
from data_prep import build_loader
from utils import create_logger, argparse_namespace, parse_option
from config import Config


class MetadataOnlyClassifier(nn.Module):
    def __init__(self, metadata_dim, num_classes):
        super().__init__()
        hidden = max(64, metadata_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, metadata_tensor):
        return self.net(metadata_tensor)


class IntermediateFusionClassifier(nn.Module):
    def __init__(self, image_model, metadata_dim, num_classes):
        super().__init__()
        self.image_model = image_model
        image_feature_dim = num_classes
        metadata_feature_dim = max(32, metadata_dim)

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, metadata_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(image_feature_dim + metadata_feature_dim, max(64, num_classes * 8)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(max(64, num_classes * 8), num_classes),
        )

    def forward(self, image_tensor, metadata_tensor):
        image_logits = self.image_model(image_tensor)
        metadata_features = self.metadata_encoder(metadata_tensor)
        fused = torch.cat([image_logits, metadata_features], dim=1)
        return self.fusion_head(fused)


class LateFusionClassifier(nn.Module):
    def __init__(self, image_model, metadata_dim, num_classes):
        super().__init__()
        self.image_model = image_model
        self.metadata_model = MetadataOnlyClassifier(metadata_dim, num_classes)

    def forward(self, image_tensor, metadata_tensor):
        image_logits = self.image_model(image_tensor)
        metadata_logits = self.metadata_model(metadata_tensor)
        return 0.5 * image_logits + 0.5 * metadata_logits


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


def _extract_batch(batch):
    if isinstance(batch, tuple):
        samples, labels = batch
        return {'image': samples, 'label': labels}
    if isinstance(batch, dict):
        return batch
    raise TypeError(f'Unsupported batch format: {type(batch)}')


def create_training_model(config, metadata_dim):
    input_mode = config.DATA.INPUT_MODE
    num_classes = config.MODEL.NUM_CLASSES

    if input_mode == 'images_only':
        return build_model(config)
    if input_mode == 'metadata_only':
        return MetadataOnlyClassifier(metadata_dim=metadata_dim, num_classes=num_classes)

    image_model = build_model(config)
    if input_mode == 'intermediate_fusion':
        return IntermediateFusionClassifier(image_model=image_model, metadata_dim=metadata_dim, num_classes=num_classes)
    if input_mode == 'late_fusion':
        return LateFusionClassifier(image_model=image_model, metadata_dim=metadata_dim, num_classes=num_classes)

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

def main(config, logger):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)
    logger.info(f"Dataset loaded: {len(dataset_train)} train images, {len(dataset_val)} validation images")
    logger.info(f"Class mapping: {dataset_train.class_to_idx}")
    if len(dataset_train) > 0:
        logger.info(f"Example training sample: {dataset_train.records[0]['image_path']}")
    if len(dataset_val) > 0:
        logger.info(f"Example validation sample: {dataset_val.records[0]['image_path']}")
    
    logger.info(f"Creating model: {config.MODEL.NAME} | mode={config.DATA.INPUT_MODE}")
    model = create_training_model(config, metadata_dim=dataset_train.metadata_dim)
    model.cuda()
    scaler = torch.amp.GradScaler('cuda', enabled=config.AMP_ENABLE)
    optimizer = create_optimizer(argparse_namespace(
        opt=config.TRAIN.OPT,
        lr=config.TRAIN.BASE_LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
        momentum=config.TRAIN.MOMENTUM,
        opt_eps=config.TRAIN.EPS,
        opt_betas=config.TRAIN.BETAS
    ), model)
    lr_scheduler, _ = create_scheduler(argparse_namespace(
        sched=config.TRAIN.SCHED,
        epochs=config.TRAIN.EPOCHS,
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
        warmup_lr=config.TRAIN.WARMUP_LR,
        min_lr=config.TRAIN.MIN_LR,
        cooldown_epochs=config.TRAIN.COOLDOWN_EPOCHS,
        decay_epochs=config.TRAIN.DECAY_EPOCHS,
        decay_rate=config.TRAIN.DECAY_RATE
    ), optimizer)


    run_name = config.EXPERIMENT_NAME or config.MODEL.NAME
    wandb_run = init_wandb(config, logger)

    # Setup metrics logging to file
    log_file = os.path.join(config.OUTPUT, f'{run_name}_metrics.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'val_precision', 'val_recall'])
        early_stopping = None

    max_accuracy = 0.0
    best_targets = None
    best_preds = None
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_loss, train_acc = train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, scaler, logger)
        
        if hasattr(lr_scheduler, 'step_update'):
            lr_scheduler.step(epoch + 1)
        else:
            lr_scheduler.step()
        
        val_acc, val_loss, val_f1, val_prec, val_recall, targets, preds = validate(config, data_loader_val, model, logger)
        
        logger.info(f"[Epoch {epoch}]: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, Precision: {val_prec:.4f}, Recall: {val_recall:.4f}")
        
        # Save metrics
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_f1, val_prec, val_recall])

        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/acc': train_acc,
                'val/loss': val_loss,
                'val/acc': val_acc,
                'val/f1': val_f1,
                'val/precision': val_prec,
                'val/recall': val_recall,
            })

        if val_acc > max_accuracy:
            max_accuracy = val_acc
            best_targets = targets
            best_preds = preds
            logger.info(f'Best performance updated at epoch {epoch} with accuracy: {max_accuracy:.2f}%')
            
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        
        
        # Early Stopping Check ONLY for DenseNet ← NEW
        if early_stopping is not None:
            if early_stopping(val_acc, epoch):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break  # ← EXIT BUT CONTINUE TO SAVE EVERYTHING

    # Save confusion matrix for the best model
    if best_targets is not None:
        cm = confusion_matrix(best_targets, best_preds)
        cm_path = os.path.join(config.OUTPUT, f'{run_name}_confusion_matrix.npy')
        np.save(cm_path, cm)
        logger.info(f"Confusion matrix for best model saved to {cm_path}")

    # Save final model checkpoint
    model_path = os.path.join(config.OUTPUT, f'{run_name}.pth')
    torch.save({
        'epoch': epoch if 'epoch' in locals() else config.TRAIN.EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'max_accuracy': max_accuracy,
        'class_to_idx': getattr(dataset_train, 'class_to_idx', None),
        'input_mode': config.DATA.INPUT_MODE,
        'metadata_dim': getattr(dataset_train, 'metadata_dim', None),
        'experiment_name': run_name,
        # Note: config not saved as it's recreated during inference using get_model_config()
    }, model_path)
    logger.info(f"Model checkpoint saved to {model_path}")

    if wandb_run is not None:
        wandb_run.summary['max_accuracy'] = max_accuracy
        wandb_run.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, scaler, logger):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for idx, raw_batch in enumerate(data_loader):
        batch = _extract_batch(raw_batch)
        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            outputs, targets = forward_by_mode(model, batch, config.DATA.INPUT_MODE)
            if config.MODEL.LABEL_SMOOTHING > 0:
                loss = torch.nn.CrossEntropyLoss(label_smoothing=config.MODEL.LABEL_SMOOTHING)(outputs, targets)
            else:
                loss = torch.nn.CrossEntropyLoss()(outputs, targets)

        scaler.scale(loss).backward()
        if config.TRAIN.CLIP_GRAD:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if hasattr(lr_scheduler, 'step_update'):
          lr_scheduler.step_update(epoch * num_steps + idx)
        
        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item(), targets.size(0))
        acc_meter.update(acc1.item(), targets.size(0))

        if idx % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch: [{epoch}][{idx}/{num_steps}] lr {lr:.6f} loss {loss_meter.avg:.4f} acc {acc_meter.avg:.2f}%')

    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def validate(config, data_loader, model, logger=None):
    model.eval()
    acc1_meter, loss_meter = AverageMeter(), AverageMeter()
    all_preds = []
    all_targets = []

    for raw_batch in data_loader:
        batch = _extract_batch(raw_batch)
        output, target = forward_by_mode(model, batch, config.DATA.INPUT_MODE)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        acc1, _ = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        preds = torch.argmax(output, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    f1 = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

    return acc1_meter.avg, loss_meter.avg, f1, precision, recall, all_targets, all_preds

if __name__ == '__main__':
    args = parse_option()
    MODEL_TO_RUN = args.model_to_run
    
    # Load model-specific config via models helper
    model_config_data = get_model_config(MODEL_TO_RUN)
    config = Config(model_config_data)
    apply_cli_overrides(config, args)

    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(str(repo_root / '.env'))
    
    torch.cuda.set_device(0)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    logger_name = config.EXPERIMENT_NAME or config.MODEL.NAME
    logger = create_logger(output_dir=config.OUTPUT, name=logger_name)
    main(config, logger)
