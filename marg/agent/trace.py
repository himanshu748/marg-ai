import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from marg.store.local import atomic_write_text


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
        return atomic_write_text(
            target,
            "".join(json.dumps(asdict(entry), ensure_ascii=False) + "\n" for entry in self.entries),
        )
