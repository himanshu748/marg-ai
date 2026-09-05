import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from marg.agent.llm import default_llm
from marg.agent.loop import run_agent
from marg.agent.tools import ToolContext
from marg.store.local import LocalStore
from marg.vision.config import VisionConfig
from marg.vision.detector import DNNDetector
from marg.vision.models import SurveyResult


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = ""


_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="marg-agent")


def create_app(data_root: Path | str) -> FastAPI:
    root = Path(data_root).resolve()
    app = FastAPI(title="MargAI Survey Dashboard")

    def survey_dir(survey_id: str) -> Path:
        candidate = (root / survey_id).resolve()
        if candidate.parent != root or not candidate.is_dir():
            raise HTTPException(status_code=404, detail="Survey not found")
        return candidate

    def load_result(survey_id: str) -> tuple[Path, SurveyResult]:
        directory = survey_dir(survey_id)
        path = directory / "result.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Survey not found")
        return directory, SurveyResult.model_validate_json(path.read_text(encoding="utf-8"))

    def decisions_path(directory: Path) -> Path:
        return directory / "agent" / "decisions.json"

    def load_decisions(directory: Path) -> dict[str, dict[str, object]]:
        path = decisions_path(directory)
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def agent_payload(directory: Path) -> dict[str, object]:
        path = directory / "agent" / "agent_result.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Agent result not found")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise HTTPException(status_code=500, detail="Invalid agent result")
        decisions = load_decisions(directory)
        work_orders = value.get("work_orders", [])
        if isinstance(work_orders, list):
            for work_order in work_orders:
                if isinstance(work_order, dict):
                    decision = decisions.get(str(work_order.get("work_order_id")))
                    if decision:
                        work_order.update(decision)
        return value

    def safe_media(survey_id: str, category: Literal["keyframes", "crops"], name: str) -> Path:
        directory = survey_dir(survey_id)
        if Path(name).name != name:
            raise HTTPException(status_code=404, detail="Media not found")
        candidate = (directory / category / name).resolve()
        base = (directory / category).resolve()
        if candidate.parent != base or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Media not found")
        return candidate

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/surveys")
    def list_surveys() -> list[dict[str, object]]:
        surveys: list[dict[str, object]] = []
        if not root.is_dir():
            return surveys
        for directory in sorted(root.iterdir()):
            result_path = directory / "result.json"
            if not directory.is_dir() or not result_path.is_file():
                continue
            result = SurveyResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            surveys.append(
                {
                    "id": directory.name,
                    "survey_id": result.survey_id,
                    "frames": result.frames,
                    "keyframes": result.keyframes,
                    "detections": result.detections,
                    "instances": len(result.instances),
                    "runtime": result.metrics.get("runtime_s", 0.0),
                    "geo_source": result.geo_source,
                }
            )
        return surveys

    @app.get("/api/surveys/{survey_id}")
    def get_survey(survey_id: str) -> dict[str, object]:
        _, result = load_result(survey_id)
        return result.model_dump(mode="json")

    @app.get("/api/surveys/{survey_id}/agent")
    def get_agent(survey_id: str) -> dict[str, object]:
        directory, _ = load_result(survey_id)
        return agent_payload(directory)

    @app.get("/api/surveys/{survey_id}/trace")
    def get_trace(survey_id: str) -> list[dict[str, object]]:
        directory, _ = load_result(survey_id)
        path = directory / "agent" / "trace.jsonl"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Agent trace not found")
        entries: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    entries.append(value)
        return entries

    @app.get("/api/surveys/{survey_id}/keyframes/{name}")
    def get_keyframe(survey_id: str, name: str) -> FileResponse:
        return FileResponse(safe_media(survey_id, "keyframes", name))

    @app.get("/api/surveys/{survey_id}/crops/{name}")
    def get_crop(survey_id: str, name: str) -> FileResponse:
        return FileResponse(safe_media(survey_id, "crops", name))

    @app.post("/api/surveys/{survey_id}/work_orders/{work_order_id}/decision")
    def decide_work_order(
        survey_id: str, work_order_id: str, request: DecisionRequest
    ) -> dict[str, object]:
        directory, _ = load_result(survey_id)
        payload = agent_payload(directory)
        work_orders = payload.get("work_orders", [])
        if not isinstance(work_orders, list):
            raise HTTPException(status_code=404, detail="Work order not found")
        order = next(
            (
                item
                for item in work_orders
                if isinstance(item, dict) and str(item.get("work_order_id")) == work_order_id
            ),
            None,
        )
        if order is None:
            raise HTTPException(status_code=404, detail="Work order not found")
        decision = {
            "decision": request.decision,
            "decision_note": request.note,
            "status": "approved" if request.decision == "approve" else "rejected",
        }
        agent_dir = directory / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        decisions = load_decisions(directory)
        decisions[work_order_id] = decision
        decisions_path(directory).write_text(json.dumps(decisions, indent=2), encoding="utf-8")
        order.update(decision)
        return order

    @app.post("/api/surveys/{survey_id}/run_agent")
    def run_agent_endpoint(
        survey_id: str, llm: Literal["mock", "bedrock"] = "mock"
    ) -> dict[str, str]:
        directory, _ = load_result(survey_id)
        _EXECUTOR.submit(_run_agent, directory, llm)
        return {"status": "started", "survey_id": survey_id, "llm": llm}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    return app


def _run_agent(directory: Path, llm_kind: str) -> None:
    result = SurveyResult.model_validate_json(
        (directory / "result.json").read_text(encoding="utf-8")
    )
    agent_dir = directory / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    config = VisionConfig()
    detector = DNNDetector(config.model_file(Path(__file__).resolve().parents[2]), config)
    context = ToolContext(
        result=result,
        store=LocalStore(agent_dir / "store"),
        detector=detector,
        keyframe_dir=directory / "keyframes",
    )
    run = run_agent(result, default_llm(llm_kind), context)
    run.trace.write_jsonl(agent_dir / "trace.jsonl")
    (agent_dir / "agent_result.json").write_text(
        json.dumps(run.to_dict(), indent=2), encoding="utf-8"
    )


app = create_app(Path("outputs"))
