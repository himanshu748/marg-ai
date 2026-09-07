import json
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO

import boto3

from marg.store.local import validate_survey_id
from marg.vision.models import SurveyResult


def _dynamo_value(value: object) -> object:
    """DynamoDB requires Decimal for the float metrics produced by the pipeline."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: _dynamo_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dynamo_value(item) for item in value]
    return value


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
        validate_survey_id(survey_id)
        if Path(suffix).is_absolute() or ".." in Path(suffix).parts:
            raise ValueError("Invalid survey object path")
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
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
        os.close(descriptor)
        try:
            self.s3.download_file(self.bucket, key, temporary)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return destination

    def list_survey_ids(self) -> list[str]:
        prefix = f"{self.prefix}/" if self.prefix else ""
        ids = []
        for response in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
            for entry in response.get("CommonPrefixes", []):
                value = str(entry.get("Prefix", "")).removeprefix(prefix).strip("/")
                if value:
                    ids.append(value)
        return sorted(ids)

    def sync_survey(self, survey_id: str) -> Path:
        validate_survey_id(survey_id)
        directory = self.cache_root / survey_id
        result_path = directory / "result.json"
        self.download_file(self.object_key(survey_id, "result.json"), result_path)
        pages = self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket,
            Prefix=self.object_key(survey_id, "agent/") + "/",
        )
        for response in pages:
            for entry in response.get("Contents", []):
                key = str(entry["Key"])
                relative = key.removeprefix(self.object_key(survey_id) + "/")
                target = (directory / relative).resolve()
                if not target.is_relative_to(directory.resolve()) or key.endswith("/"):
                    continue
                self.download_file(key, target)
        return directory

    def ensure_media(self, survey_id: str, category: str, name: str) -> Path:
        validate_survey_id(survey_id)
        if category not in {"keyframes", "crops", "evidence", "agent_crops"} or Path(name).name != name:
            raise ValueError("Invalid media path")
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
            Item=_dynamo_value({
                "survey_id": survey_id,
                "work_order_id": work_order_id,
                **value,
            })
        )

    def save_decision(
        self,
        survey_id: str,
        work_order_id: str,
        decision: dict[str, object],
    ) -> dict[str, object]:
        if decision.get("decision") not in {"approve", "reject"}:
            raise ValueError("Invalid work-order decision")
        expected_status = "approved" if decision["decision"] == "approve" else "rejected"
        if decision.get("status") != expected_status:
            raise ValueError("Decision status does not match decision")
        if not isinstance(decision.get("agent_run_id"), str) or not decision["agent_run_id"]:
            raise ValueError("A completed agent run is required to decide a work order")
        response = self.dynamodb.Table(self.work_orders_table).update_item(
            Key={"survey_id": validate_survey_id(survey_id), "work_order_id": work_order_id},
            UpdateExpression=(
                "SET #status = :status, #decision = :decision, "
                "decision_note = if_not_exists(decision_note, :note), "
                "decided_at = if_not_exists(decided_at, :decided_at)"
            ),
            ConditionExpression=(
                "#run = :run AND (#status = :pending OR "
                "(#status = :status AND #decision = :decision))"
            ),
            ExpressionAttributeNames={"#status": "status", "#decision": "decision", "#run": "agent_run_id"},
            ExpressionAttributeValues={
                ":status": decision["status"],
                ":decision": decision["decision"],
                ":note": decision["decision_note"],
                ":decided_at": decision["decided_at"],
                ":run": decision["agent_run_id"],
                ":pending": "pending_approval",
            },
            ReturnValues="ALL_NEW",
        )
        # An idempotent retry retains the original decision timestamp and note.
        saved = response.get("Attributes", {})
        return {key: saved.get(key, value) for key, value in decision.items()}

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
        validate_survey_id(survey_id)
        values.pop("survey_id", None)
        item: dict[str, object] = {"status": status, **values, "updated_at": datetime.now(timezone.utc).isoformat()}
        # Preserve upload keys and creation metadata across processing/retry updates.
        self.dynamodb.Table(self.surveys_table).update_item(
            Key={"survey_id": survey_id},
            UpdateExpression="SET " + ", ".join(f"#v{index} = :v{index}" for index in range(len(item))),
            ExpressionAttributeNames={f"#v{index}": key for index, key in enumerate(item)},
            ExpressionAttributeValues={f":v{index}": _dynamo_value(value) for index, value in enumerate(item.values())},
        )

    def survey_status(self, survey_id: str) -> dict[str, object]:
        response = self.dynamodb.Table(self.surveys_table).get_item(
            Key={"survey_id": validate_survey_id(survey_id)}, ConsistentRead=True
        )
        value = response.get("Item", {})
        return value if isinstance(value, dict) else {}

    def list_statuses(self) -> list[dict[str, object]]:
        table = self.dynamodb.Table(self.surveys_table)
        response = table.scan()
        values = response.get("Items", [])
        while response.get("LastEvaluatedKey"):
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            values.extend(response.get("Items", []))
        return [value for value in values if isinstance(value, dict)]
