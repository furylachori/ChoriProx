import json
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict
    source: str = "llm"
    from_cache: bool = False  # M3-2: set by VFS cache-hit path to bypass read-once guard

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("ToolCall name cannot be empty")

    @classmethod
    def from_openai(cls, tc: dict) -> "ToolCall":
        func = tc["function"]
        args = func["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        return cls(id=tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                   name=func["name"], arguments=args, source="llm")

    @classmethod
    def from_parsed(cls, name: str, arguments: dict, existing_id: str = None) -> "ToolCall":
        return cls(id=existing_id or f"call_{uuid.uuid4().hex[:24]}",
                   name=name, arguments=arguments, source="parsed")

    def to_openai(self) -> dict:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": json.dumps(self.arguments)}}
