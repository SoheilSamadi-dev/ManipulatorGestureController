from datetime import datetime
from pathlib import Path
from typing import Union


class EventLogger:
    def __init__(self, file_path: Union[str, Path] = "recognized_gestures.txt"):
        self.path = Path(file_path)
        # Ensure parent exists (no-op if already exists)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, label: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{ts}] {label}\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)

