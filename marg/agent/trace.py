import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TraceEntry:
    step: int
    tool: str
    input: dict[str, object]
    output: dict[str, object]
    latency_ms: float
    model_text: str = ""


@dataclass(slots=True)
class Trace:
    entries: list[TraceEntry] = field(default_factory=list)

    def append(
        self,
        tool: str,
        input_data: dict[str, object],
        output: dict[str, object],
        latency_ms: float,
        model_text: str = "",
    ) -> TraceEntry:
        entry = TraceEntry(
            step=len(self.entries),
            tool=tool,
            input=input_data,
            output=output,
            latency_ms=latency_ms,
            model_text=model_text,
        )
        self.entries.append(entry)
        return entry

    def write_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return target
