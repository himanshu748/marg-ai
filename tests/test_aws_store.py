from marg.store.aws import S3SurveyStore


def test_s3_survey_store_key_layout() -> None:
    store = S3SurveyStore.__new__(S3SurveyStore)
    store.prefix = "surveys"
    assert store.object_key("demo", "result.json") == "surveys/demo/result.json"
    assert store.object_key("demo", "keyframes/kf_00001.jpg") == (
        "surveys/demo/keyframes/kf_00001.jpg"
    )

    store.prefix = ""
    assert store.object_key("demo", "agent/agent_result.json") == (
        "demo/agent/agent_result.json"
    )


def test_status_update_preserves_upload_metadata_and_serializes_float_metrics():
    from decimal import Decimal

    class Table:
        def __init__(self):
            self.item = {"survey_id": "demo", "video_key": "incoming/video.mp4", "created_at": "original"}

        def update_item(self, **kwargs):
            for alias, key in kwargs["ExpressionAttributeNames"].items():
                self.item[key] = kwargs["ExpressionAttributeValues"][":" + alias[1:]]

        def get_item(self, **kwargs):
            assert kwargs["ConsistentRead"] is True
            return {"Item": self.item.copy()}

    class Dynamo:
        def __init__(self):
            self.table = Table()

        def Table(self, name):
            return self.table

    store = S3SurveyStore.__new__(S3SurveyStore)
    store.surveys_table = "surveys"
    store.dynamodb = Dynamo()
    store.update_survey_status("demo", "processing")
    store.update_survey_status("demo", "done", runtime=1.25, nested={"confidence": 0.5})
    status = store.survey_status("demo")
    assert status["video_key"] == "incoming/video.mp4"
    assert status["created_at"] == "original"
    assert status["runtime"] == Decimal("1.25")
    assert status["nested"]["confidence"] == Decimal("0.5")
    assert status["updated_at"]


def test_failed_cache_download_does_not_replace_valid_result(tmp_path):
    import pytest

    class S3:
        def download_file(self, bucket, key, destination):
            from pathlib import Path
            Path(destination).write_text("partial")
            raise OSError("download interrupted")

    store = S3SurveyStore.__new__(S3SurveyStore)
    store.bucket = "bucket"
    store.s3 = S3()
    result = tmp_path / "result.json"
    result.write_text("complete")
    with pytest.raises(OSError):
        store.download_file("demo/result.json", result)
    assert result.read_text() == "complete"
    assert list(tmp_path.iterdir()) == [result]


def test_status_scan_reads_all_pages():
    class Table:
        def scan(self, **kwargs):
            if kwargs.get("ExclusiveStartKey"):
                return {"Items": [{"survey_id": "second"}]}
            return {"Items": [{"survey_id": "first"}], "LastEvaluatedKey": {"survey_id": "first"}}

    class Dynamo:
        def Table(self, name):
            return Table()

    store = S3SurveyStore.__new__(S3SurveyStore)
    store.surveys_table = "surveys"
    store.dynamodb = Dynamo()
    assert [row["survey_id"] for row in store.list_statuses()] == ["first", "second"]


def test_decision_is_conditioned_on_run_and_pending_state_and_retains_original_record():
    recorded = []
    decision = {
        "decision": "approve", "status": "approved", "decision_note": "second note",
        "decided_at": "second time", "agent_run_id": "run-current",
    }

    class Table:
        def update_item(self, **kwargs):
            recorded.append(kwargs)
            return {"Attributes": {**decision, "decision_note": "original note", "decided_at": "original time"}}

    class Dynamo:
        def Table(self, name):
            return Table()

    store = S3SurveyStore.__new__(S3SurveyStore)
    store.work_orders_table = "work-orders"
    store.dynamodb = Dynamo()
    saved = store.save_decision("demo", "order-1", decision)
    request = recorded[0]
    assert request["ConditionExpression"] == "#run = :run AND (#status = :pending OR (#status = :status AND #decision = :decision))"
    assert request["ExpressionAttributeValues"][":run"] == "run-current"
    assert request["ExpressionAttributeValues"][":pending"] == "pending_approval"
    assert request["ReturnValues"] == "ALL_NEW"
    assert "if_not_exists(decided_at, :decided_at)" in request["UpdateExpression"]
    assert saved["decision_note"] == "original note"
    assert saved["decided_at"] == "original time"
    assert saved["agent_run_id"] == "run-current"


def test_stale_cloud_decision_conflict_is_propagated():
    import pytest
    from botocore.exceptions import ClientError

    class Table:
        def update_item(self, **kwargs):
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException", "Message": "State changed"}}, "UpdateItem")

    class Dynamo:
        def Table(self, name):
            return Table()

    store = S3SurveyStore.__new__(S3SurveyStore)
    store.work_orders_table = "work-orders"
    store.dynamodb = Dynamo()
    decision = {"decision": "reject", "status": "rejected", "decision_note": "reviewed", "decided_at": "now", "agent_run_id": "run-stale"}
    with pytest.raises(ClientError) as failure:
        store.save_decision("demo", "order-1", decision)
    assert failure.value.response["Error"]["Code"] == "ConditionalCheckFailedException"
    with pytest.raises(ValueError, match="completed agent run"):
        store.save_decision("demo", "order-1", {**decision, "agent_run_id": None})
