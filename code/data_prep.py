import os
import csv
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def _to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == '':
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


class MetadataVectorizer:
    def __init__(self, metadata_keys, numeric_stats, categorical_maps):
        self.metadata_keys = metadata_keys
        self.numeric_stats = numeric_stats
        self.categorical_maps = categorical_maps
        self.output_dim = len(metadata_keys)

    @classmethod
    def fit(cls, records):
        metadata_keys = sorted({k for r in records for k in r['metadata'].keys()})
        numeric_stats = {}
        categorical_maps = {}

        for key in metadata_keys:
            float_values = []
            raw_values = []
            for record in records:
                value = record['metadata'].get(key)
                raw_values.append('' if value is None else str(value).strip())
                fv = _to_float(value)
                if fv is not None:
                    float_values.append(fv)

            is_numeric = len(float_values) >= max(1, int(0.8 * len(records)))
            if is_numeric:
                mean = sum(float_values) / max(1, len(float_values))
                var = sum((v - mean) ** 2 for v in float_values) / max(1, len(float_values))
                std = var ** 0.5
                if std == 0:
                    std = 1.0
                numeric_stats[key] = (mean, std)
            else:
                uniq = sorted({v for v in raw_values if v != ''})
                categorical_maps[key] = {v: i for i, v in enumerate(uniq)}

        return cls(metadata_keys, numeric_stats, categorical_maps)

    def transform(self, metadata):
        values = []
        for key in self.metadata_keys:
            raw_value = metadata.get(key)
            if key in self.numeric_stats:
                mean, std = self.numeric_stats[key]
                fv = _to_float(raw_value)
                if fv is None:
                    fv = mean
                values.append((fv - mean) / std)
            else:
                value = '' if raw_value is None else str(raw_value).strip()
                mapping = self.categorical_maps.get(key, {})
                idx = mapping.get(value, -1)
                denom = max(1, len(mapping))
                values.append(float(idx + 1) / float(denom))
        if not values:
            values = [0.0]
        return torch.tensor(values, dtype=torch.float32)


class MetadataImageDataset(Dataset):
    def __init__(self, records, class_to_idx, transform=None, return_metadata=False, input_mode='images_only', metadata_vectorizer=None):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_metadata = return_metadata
        self.input_mode = input_mode
        self.metadata_vectorizer = metadata_vectorizer
        self.metadata_dim = metadata_vectorizer.output_dim if metadata_vectorizer else 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = None
        if record.get('image_path'):
            image = Image.open(record['image_path']).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        label = self.class_to_idx[record['label']]

        metadata_tensor = self.metadata_vectorizer.transform(record['metadata']) if self.metadata_vectorizer else torch.tensor([0.0], dtype=torch.float32)

        input_mode = self.input_mode
        if input_mode == 'images_only':
            return image, label
        if input_mode == 'metadata_only':
            return {
                'label': label,
                'metadata': record['metadata'],
                'metadata_tensor': metadata_tensor,
                'image_path': record.get('image_path'),
            }

        if input_mode in {'intermediate_fusion', 'late_fusion'} or self.return_metadata:
            return {
                'image': image,
                'label': label,
                'metadata': record['metadata'],
                'metadata_tensor': metadata_tensor,
                'image_path': record['image_path'],
            }
        return image, label


def _resolve_metadata_path(config):
    metadata_candidates = [
        config.DATA.CLEANED_METADATA_PATH,
        config.DATA.METADATA_PATH,
        os.path.join(config.DATA.DATA_PATH, 'metadata_cleaned.csv'),
        os.path.join(config.DATA.DATA_PATH, 'metadata.csv'),
    ]

    for candidate in metadata_candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find metadata CSV. Checked: {metadata_candidates}"
    )


def _resolve_images_dir(config):
    candidates = [
        config.DATA.IMAGES_DIR,
        os.path.join(config.DATA.DATA_PATH, 'images'),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find images directory. Checked: {candidates}")


def _infer_column(fieldnames, configured_name, candidates):
    if configured_name:
        if configured_name not in fieldnames:
            raise ValueError(f"Configured column '{configured_name}' not present in metadata")
        return configured_name
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _resolve_image_path(images_dir, image_value):
    image_value = (image_value or '').strip()
    if not image_value:
        return None

    normalized = image_value.replace('\\', '/')
    filename = os.path.basename(normalized)

    candidates = [
        os.path.join(images_dir, normalized),
        os.path.join(images_dir, filename),
    ]

    stem, ext = os.path.splitext(filename)
    if not ext:
        for candidate_ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            candidates.append(os.path.join(images_dir, stem + candidate_ext))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _read_records(config):
    metadata_path = _resolve_metadata_path(config)
    input_mode = getattr(config.DATA, 'INPUT_MODE', 'images_only')
    images_dir = None
    if input_mode != 'metadata_only':
        images_dir = _resolve_images_dir(config)

    with open(metadata_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        image_col = _infer_column(
            fieldnames,
            config.DATA.IMAGE_COLUMN,
            ['image_id', 'image', 'img_id', 'img', 'path', 'filename'],
        )
        label_col = _infer_column(
            fieldnames,
            config.DATA.LABEL_COLUMN,
            ['diagnostic', 'label', 'target', 'class'],
        )
        split_col = _infer_column(
            fieldnames,
            config.DATA.SPLIT_COLUMN,
            ['split', 'set', 'subset', 'fold'],
        )

        if label_col is None:
            raise ValueError(
                f"Metadata requires a label column. Found columns: {fieldnames}"
            )
        if input_mode != 'metadata_only' and image_col is None:
            raise ValueError(
                f"Image-based modes require an image column. Found columns: {fieldnames}"
            )

        records = []
        for row in reader:
            image_value = row.get(image_col) if image_col else ''
            image_path = _resolve_image_path(images_dir, image_value) if images_dir else (image_value or '').strip()
            label = (row.get(label_col) or '').strip()
            if (input_mode != 'metadata_only' and image_path is None) or not label:
                continue

            metadata = {
                k: v
                for k, v in row.items()
                if k != label_col and (image_col is None or k != image_col)
            }
            split_value = (row.get(split_col) or '').strip().lower() if split_col else ''

            records.append({
                'image_path': image_path,
                'label': label,
                'split': split_value,
                'metadata': metadata,
            })

    if not records:
        raise ValueError(
            f"No usable samples were found using metadata '{metadata_path}' and images '{images_dir}'"
        )

    return records


def _split_records(records, config):
    has_split = any(r['split'] for r in records)
    if has_split:
        train_aliases = {'train', 'training', 'tr'}
        val_aliases = {'val', 'valid', 'validation', 'dev'}
        train_records = [r for r in records if r['split'] in train_aliases]
        eval_records = [r for r in records if r['split'] in val_aliases]

        if train_records and eval_records:
            return train_records, eval_records

    rng = random.Random(config.DATA.SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)

    val_ratio = float(config.DATA.VAL_SPLIT)
    if len(shuffled) == 1:
        return shuffled, shuffled

    val_size = max(1, int(len(shuffled) * val_ratio))
    val_size = min(val_size, len(shuffled) - 1)
    eval_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    return train_records, eval_records

def build_loader(config):

    input_mode = getattr(config.DATA, 'INPUT_MODE', 'images_only')

    train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),          
    transforms.RandomHorizontalFlip(),   
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    records = _read_records(config)
    class_names = sorted({record['label'] for record in records})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_records, eval_records = _split_records(records, config)
    metadata_vectorizer = MetadataVectorizer.fit(train_records)

    train_dataset = MetadataImageDataset(
        train_records,
        class_to_idx,
        transform=train_transform if input_mode != 'metadata_only' else None,
        return_metadata=config.DATA.RETURN_METADATA,
        input_mode=input_mode,
        metadata_vectorizer=metadata_vectorizer,
    )
    eval_dataset = MetadataImageDataset(
        eval_records,
        class_to_idx,
        transform=eval_transform if input_mode != 'metadata_only' else None,
        return_metadata=config.DATA.RETURN_METADATA,
        input_mode=input_mode,
        metadata_vectorizer=metadata_vectorizer,
    )

    train_dataset.classes = class_names
    train_dataset.class_to_idx = class_to_idx
    eval_dataset.classes = class_names
    eval_dataset.class_to_idx = class_to_idx

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY,
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY,
    )

    return train_dataset, eval_dataset, train_loader, eval_loader
