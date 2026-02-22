import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
BATCH_DIR = DATA_DIR / "cifar-10-batches-py"
