"""Optional DeepSeek structured-output adapter. Requires DEEPSEEK_API_KEY."""
import asyncio
import os
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel

from .agents import StructuredModel
from .models import UsageSummary


class StructuredOutputError(RuntimeError):
    """Raised when a provider response cannot be parsed as the requested schema."""


@dataclass
class TokenBudget:
    """Shares a capped generated-token budget safely across concurrent agents."""
    max_output_tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    reserved_output_tokens: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def reserve(self, requested_tokens: int) -> int:
        async with self.lock:
            remaining = self.max_output_tokens - self.output_tokens - self.reserved_output_tokens
            if remaining <= 0:
                raise RuntimeError("The run output-token budget has been reached.")
            allocation = min(requested_tokens, remaining)
            self.reserved_output_tokens += allocation
            return allocation

    async def release(self, reservation: int) -> None:
        async with self.lock:
            self.reserved_output_tokens -= reservation

    async def record(self, response, reservation: int) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
        async with self.lock:
            self.reserved_output_tokens -= reservation
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.calls += 1

    def summary(self) -> UsageSummary:
        return UsageSummary(input_tokens=self.input_tokens, output_tokens=self.output_tokens, total_tokens=self.input_tokens + self.output_tokens, calls=self.calls, max_output_tokens=self.max_output_tokens)


class DeepSeekResponsesModel(StructuredModel):
    def __init__(self, model: str, api_key: str | None = None, max_run_output_tokens: int = 8_000, usage_callback: Callable[[UsageSummary], None] | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.usage_callback = usage_callback
        self.budget = TokenBudget(max_output_tokens=max_run_output_tokens)

    async def generate(self, *, instructions: str, input_text: str, output_type: type[BaseModel]) -> BaseModel:
        from openai import AsyncOpenAI

        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for the DeepSeek provider.")
        client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        last_error = "unknown provider error"
        requested_tokens = self._target_output_limit(output_type.__name__)
        # DeepSeek documents that JSON mode can occasionally return empty content.
        # Retry once and retain diagnostics instead of surfacing a vague Pydantic error.
        for attempt in range(1, 3):
            reservation = await self.budget.reserve(requested_tokens)
            try:
                response = await client.responses.create(
                    model=self.model,
                    instructions=instructions + "\nReturn only a complete valid JSON object matching the requested schema.",
                    input=input_text,
                    max_output_tokens=reservation,
                    reasoning={"effort": "none"},
                    temperature=0,
                    text={"format": {"type": "json_schema", "name": output_type.__name__, "schema": output_type.model_json_schema()}},
                )
            except Exception:
                await self.budget.release(reservation)
                raise
            await self.budget.record(response, reservation)
            if self.usage_callback:
                self.usage_callback(self.budget.summary())
            raw_output = self._output_text(response)
            try:
                if not raw_output.strip():
                    status = getattr(response, "status", "unknown")
                    detail = getattr(response, "incomplete_details", None)
                    raise StructuredOutputError(f"DeepSeek returned an empty response for {output_type.__name__} (status={status}, incomplete_details={detail}).")
                return output_type.model_validate_json(raw_output)
            except (ValueError, StructuredOutputError) as error:
                last_error = f"attempt {attempt}/2: {error}"
        raise StructuredOutputError(f"Could not obtain valid JSON for {output_type.__name__}; {last_error}")

    @staticmethod
    def _output_text(response) -> str:
        """Use the convenience field, with a fallback for provider SDK variations."""
        text = getattr(response, "output_text", "") or ""
        if text:
            return text
        parts = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    parts.append(getattr(content, "text", ""))
        return "".join(parts)

    def usage_summary(self) -> UsageSummary:
        return self.budget.summary()

    @staticmethod
    def _target_output_limit(schema_name: str) -> int:
        """Role-aware caps avoid starving a large test-case response.

        The shared run budget remains the hard ceiling; these are only the
        maximum reservation each specialist may request.
        """
        return {
            "RequirementList": 4_000,
            "TextList": 1_200,
            "GeneratedCases": 6_000,
            "Critique": 4_000,
        }.get(schema_name, 1_500)
