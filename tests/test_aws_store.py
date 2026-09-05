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
