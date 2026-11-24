import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


def get_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else BASE_DIR / path


APP_NAME = os.getenv("APP_NAME", "PlagiatGuard")
JWT_SECRET = os.getenv("JWT_SECRET", "unsafe-development-secret")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
DAILY_QUOTA = int(os.getenv("DAILY_QUOTA", "20"))

DATABASE_PATH = get_path("DATABASE_PATH", "data/plagiatguard.db")
CORPUS_DIR = get_path("CORPUS_DIR", "data/corpus")
REPORTS_DIR = get_path("REPORTS_DIR", "reports")
FAISS_INDEX_PATH = get_path("FAISS_INDEX_PATH", "data/faiss.index")
FAISS_META_PATH = get_path("FAISS_META_PATH", "data/faiss_meta.json")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
ENABLE_WEB_SEARCH = (
    os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
)
USE_EMBEDDINGS = (
    os.getenv("USE_EMBEDDINGS", "false").lower() == "true"
)

for directory in (
    DATABASE_PATH.parent,
    CORPUS_DIR,
    REPORTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)