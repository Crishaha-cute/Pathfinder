from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


FEEDBACK_FIELDS = ["submitted_at", "rating", "message", "name", "email"]


def save_feedback(path: Path, rating: int, message: str, name: str = "", email: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FEEDBACK_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "rating": rating,
                "message": message.strip(),
                "name": name.strip(),
                "email": email.strip(),
            }
        )
