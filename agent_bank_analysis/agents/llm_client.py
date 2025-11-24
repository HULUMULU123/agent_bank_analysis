"""LLM abstraction with a deterministic fallback client."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List


@dataclass
class LLMResponse:
    raw: str


class LLMClient:
    """Abstract interface for language model calls."""

    def generate(self, prompt: str) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def parse_json_list(response: LLMResponse) -> List[dict[str, Any]]:
        try:
            payload = json.loads(response.raw)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and "items" in payload:
                return payload["items"]  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass
        return []


class EchoLLMClient(LLMClient):
    """Deterministic client used as a placeholder during development."""

    def generate(self, prompt: str) -> LLMResponse:
        # Extract txn_ids to create a synthetic structured response
        items: List[dict[str, Any]] = []
        for line in prompt.splitlines():
            if "txn_id=" in line:
                try:
                    txn_id = line.split("txn_id=")[1].split(",")[0].strip()
                except IndexError:
                    continue
                items.append(
                    {
                        "txn_id": txn_id,
                        "risk_level": "yellow",
                        "risk": 0.35,
                        "recommendation": "Провести дополнительную проверку",
                        "reasoning": "Заглушка: требуется подключение боевого LLM.",
                    }
                )
        return LLMResponse(raw=json.dumps(items, ensure_ascii=False))
