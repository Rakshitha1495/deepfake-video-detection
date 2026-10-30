"""Backend settings, read from environment variables or the project's .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.config import ARTIFACTS_DIR, CHECKPOINT_PATH, PROJECT_ROOT, VIDEO_EXTENSIONS


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def _origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


def _checkpoint_path() -> Path:
    p = Path(os.getenv("MODEL_PATH", str(CHECKPOINT_PATH)))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    checkpoint_path: Path = field(default_factory=_checkpoint_path)
    artifacts_dir: Path = ARTIFACTS_DIR
    device: str | None = os.getenv("DEVICE") or None  # "cpu", "cuda" or empty for auto
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "100"))
    allowed_extensions: frozenset = frozenset(VIDEO_EXTENSIONS)
    allowed_origins: list[str] = field(default_factory=_origins)
    torch_threads: int = int(os.getenv("TORCH_THREADS", "0"))  # 0 = PyTorch default

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
