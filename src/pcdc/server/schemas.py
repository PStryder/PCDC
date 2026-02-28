"""OpenAI-compatible Pydantic models for the PCDC chat API."""

from __future__ import annotations

import time
from typing import Any

from typing import Literal

from pydantic import BaseModel, Field


# --- Request ---

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "pcdc"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | str | None = None


# --- PCDC metadata (extra field, ignored by standard frontends) ---

class PCDCMetadata(BaseModel):
    settling_energy: float          # blended energy (used for temperature)
    reconstruction_energy: float    # Phase 1 energy
    predictive_energy: float        # Phase 2 energy
    converged: bool
    adjusted_temperature: float
    settle_steps: int
    cosine_distance: float | None = None  # embedding distance to previous turn
    retrieval_triggered: bool = False
    retrieval_count: int = 0
    deviation_match_score: float | None = None


# --- Non-streaming response ---

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "pcdc"
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    pcdc: PCDCMetadata | None = None


# --- Streaming response ---

class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "pcdc"
    choices: list[ChatCompletionChunkChoice]
    pcdc: PCDCMetadata | None = None
