import json
import os
import re
import tempfile
from pathlib import Path

from marg.vision.models import SurveyResult


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Replace a complete file on the same filesystem; readers never see half JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: object) -> Path:
    return atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def validate_survey_id(survey_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", survey_id):
        raise ValueError("Invalid survey identifier")
    return survey_id


class LocalStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_survey(self, result: SurveyResult) -> Path:
        directory = self.root / "survey" / validate_survey_id(result.survey_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "result.json"
        atomic_write_text(path, result.model_dump_json(indent=2))
        return path

    def load_survey(self, survey_id: str) -> SurveyResult:
        path = self.root / "survey" / validate_survey_id(survey_id) / "result.json"
        return SurveyResult.model_validate_json(path.read_text(encoding="utf-8"))

    def save_json(self, survey_id: str, name: str, value: dict[str, object]) -> Path:
        directory = self.root / "survey" / validate_survey_id(survey_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        if path.resolve().parent != directory.resolve():
            raise ValueError("Invalid store filename")
        atomic_write_json(path, value)
        return path

    def save_work_order(self, survey_id: str, work_order_id: str, value: dict[str, object]) -> Path:
        return self.save_json(survey_id, f"work_order_{work_order_id}.json", value)

    def save_resurvey(self, survey_id: str, resurvey_id: str, value: dict[str, object]) -> Path:
        return self.save_json(survey_id, f"resurvey_{resurvey_id}.json", value)

    def save_dismissal(self, survey_id: str, instance_id: int, value: dict[str, object]) -> Path:
        return self.save_json(survey_id, f"dismissal_{instance_id}.json", value)
