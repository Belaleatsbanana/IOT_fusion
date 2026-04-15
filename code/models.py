from CNN.CNN import build_cnn
from Transformer.Transformer import build_transformer

def get_model_config(model_name):
    if model_name == 'cnn':
        from CNN.config import get_config
        return get_config()
    elif model_name == 'transformer':
        from Transformer.config import get_config
        return get_config()
    raise ValueError(f'Unsupported model_to_run: {model_name}. Expected one of: cnn, transformer')

def build_model(config):
    model_name = config.MODEL.NAME

    # Extract all parameters from config.MODEL to pass to the model builder
    # We lowercase the keys to match Python argument naming conventions
    model_kwargs = {k.lower(): v for k, v in config.MODEL.__dict__.items() if k != 'NAME' and not k.startswith('_')}

    if model_name == 'cnn':
        return build_cnn(**model_kwargs)
    if model_name == 'transformer':
        return build_transformer(**model_kwargs)

    raise ValueError(f'Unsupported MODEL.NAME: {model_name}. Expected one of: cnn, transformer')
