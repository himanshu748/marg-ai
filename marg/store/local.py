import json
from pathlib import Path

from marg.vision.models import SurveyResult


class LocalStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_survey(self, result: SurveyResult) -> Path:
        directory = self.root / "survey" / result.survey_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "result.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_survey(self, survey_id: str) -> SurveyResult:
        path = self.root / "survey" / survey_id / "result.json"
        return SurveyResult.model_validate_json(path.read_text(encoding="utf-8"))

    def save_json(self, survey_id: str, name: str, value: dict[str, object]) -> Path:
        directory = self.root / "survey" / survey_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path
