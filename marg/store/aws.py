import json
import os
from pathlib import Path
from typing import BinaryIO

import boto3

from marg.vision.models import SurveyResult


class S3SurveyStore:
    """S3-backed survey files with a local cache."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        cache_root: str | Path = "/tmp/marg-cache",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache_root = Path(cache_root)
        self.s3 = boto3.client("s3")
        self.dynamodb = boto3.resource("dynamodb")
        self.work_orders_table = os.environ.get(
            "MARG_WORK_ORDERS_TABLE", "margai-work-orders"
        )
        self.surveys_table = os.environ.get("MARG_SURVEYS_TABLE", "margai-surveys")

    def object_key(self, survey_id: str, suffix: str = "") -> str:
        parts = [part for part in (self.prefix, survey_id, suffix.strip("/")) if part]
        return "/".join(parts)

    def upload_file(self, path: str | Path, key: str) -> None:
        self.s3.upload_file(str(path), self.bucket, key)

    def upload_fileobj(self, fileobj: BinaryIO, key: str, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else None
        self.s3.upload_fileobj(fileobj, self.bucket, key, ExtraArgs=extra)

    def download_file(self, key: str, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, key, str(destination))
        return destination

    def list_survey_ids(self) -> list[str]:
        prefix = f"{self.prefix}/" if self.prefix else ""
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, Delimiter="/")
        ids = []
        for entry in response.get("CommonPrefixes", []):
            value = str(entry.get("Prefix", "")).removeprefix(prefix).strip("/")
            if value:
                ids.append(value)
        return sorted(ids)

    def sync_survey(self, survey_id: str) -> Path:
        directory = self.cache_root / survey_id
        result_path = directory / "result.json"
        self.download_file(self.object_key(survey_id, "result.json"), result_path)
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.object_key(survey_id, "agent/") + "/",
        )
        for entry in response.get("Contents", []):
            key = str(entry["Key"])
            relative = key.removeprefix(self.object_key(survey_id) + "/")
            self.download_file(key, directory / relative)
        return directory

    def ensure_media(self, survey_id: str, category: str, name: str) -> Path:
        path = self.cache_root / survey_id / category / name
        if not path.is_file():
            self.download_file(self.object_key(survey_id, f"{category}/{name}"), path)
        return path

    def upload_directory(self, survey_id: str, directory: str | Path) -> None:
        root = Path(directory)
        for path in root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                self.upload_file(path, self.object_key(survey_id, relative))

    def save_survey(self, result: SurveyResult) -> None:
        payload = result.model_dump_json(indent=2).encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.object_key(result.survey_id, "result.json"),
            Body=payload,
            ContentType="application/json",
        )

    def load_survey(self, survey_id: str) -> SurveyResult:
        path = self.sync_survey(survey_id) / "result.json"
        return SurveyResult.model_validate_json(path.read_text(encoding="utf-8"))

    def save_json(self, survey_id: str, name: str, value: dict[str, object]) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.object_key(survey_id, name),
            Body=json.dumps(value, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def save_work_order(
        self, survey_id: str, work_order_id: str, value: dict[str, object]
    ) -> None:
        self.save_json(survey_id, f"agent/work_order_{work_order_id}.json", value)
        self.dynamodb.Table(self.work_orders_table).put_item(
            Item={
                "survey_id": survey_id,
                "work_order_id": work_order_id,
                **value,
            }
        )

    def save_decision(
        self,
        survey_id: str,
        work_order_id: str,
        decision: dict[str, object],
    ) -> None:
        self.dynamodb.Table(self.work_orders_table).update_item(
            Key={"survey_id": survey_id, "work_order_id": work_order_id},
            UpdateExpression="SET #status = :status, decision = :decision, decision_note = :note",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": decision["status"],
                ":decision": decision["decision"],
                ":note": decision["decision_note"],
            },
        )

    def save_resurvey(
        self, survey_id: str, resurvey_id: str, value: dict[str, object]
    ) -> None:
        self.save_json(survey_id, f"agent/resurvey_{resurvey_id}.json", value)

    def save_dismissal(
        self, survey_id: str, instance_id: int, value: dict[str, object]
    ) -> None:
        self.save_json(survey_id, f"agent/dismissal_{instance_id}.json", value)

    def update_survey_status(
        self,
        survey_id: str,
        status: str,
        **values: object,
    ) -> None:
        item: dict[str, object] = {"survey_id": survey_id, "status": status, **values}
        self.dynamodb.Table(self.surveys_table).put_item(Item=item)

    def survey_status(self, survey_id: str) -> dict[str, object]:
        response = self.dynamodb.Table(self.surveys_table).get_item(
            Key={"survey_id": survey_id}
        )
        value = response.get("Item", {})
        return value if isinstance(value, dict) else {}

    def list_statuses(self) -> list[dict[str, object]]:
        response = self.dynamodb.Table(self.surveys_table).scan()
        values = response.get("Items", [])
        return [value for value in values if isinstance(value, dict)]
