import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


MODEL_DIR = _env_path("MODEL_DIR", "model")
FINETUNED_MODEL_DIR = _env_path("FINETUNED_MODEL_DIR", "model_finetuned")
DB_PATH = _env_path("DB_PATH", "data/purchases.db")
EXPORT_DIR = _env_path("EXPORT_DIR", "exported_data")
MLFLOW_DB_PATH = _env_path("MLFLOW_DB_PATH", "mlflow.db")
MLFLOW_ARTIFACTS_DIR = _env_path("MLFLOW_ARTIFACTS_DIR", "mlruns")

DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "NER_Receipt_FineTuning")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
MAX_LEN = int(os.environ.get("MAX_LEN", "128"))
FIX_MISTRAL_REGEX = os.environ.get("FIX_MISTRAL_REGEX", "true").lower() in {"1", "true", "yes"}
