from pathlib import Path
import os

PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR = PARENT_DIR / 'data' / 'raw'
TRANSFORMED_DATA_DIR = PARENT_DIR / 'data' / 'transformed'
EXISTING_CHARS_DIR = PARENT_DIR / 'data' / 'chars'
DATA_CACHE_DIR = PARENT_DIR / 'data' / 'cache'

MODELS_DIR = PARENT_DIR / 'models'

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR)

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)

if not Path(DATA_CACHE_DIR).exists():
    os.mkdir(DATA_CACHE_DIR)

if not Path(EXISTING_CHARS_DIR).exists():
    os.mkdir(EXISTING_CHARS_DIR)