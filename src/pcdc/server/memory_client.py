"""Lightweight sync client for MemoryGate MCP JSON-RPC."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class MemoryResult:
    """A single result from MemoryGate search."""

    text: str
    source_type: str
    confidence: float


class MemoryClient:
    """Sync MemoryGate client via MCP JSON-RPC over HTTP.

    All failures are caught and logged — ``search()`` always returns
    a list (possibly empty) so the caller never needs error handling.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/mcp",
        timeout: float = 2.0,
        bearer_token: str | None = None,
    ):
        headers = {"Content-Type": "application/json"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        self._base_url = base_url
        self._client = httpx.Client(headers=headers, timeout=timeout)
        self._req_id = 0

    def search(
        self,
        query: str,
        limit: int = 3,
        min_confidence: float = 0.5,
        domain: str | None = None,
    ) -> list[MemoryResult]:
        """Search MemoryGate. Returns empty list on any failure."""
        self._req_id += 1
        arguments: dict = {
            "query": query,
            "limit": limit,
            "min_confidence": min_confidence,
        }
        if domain:
            arguments["domain"] = domain

        payload = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": "tools/call",
            "params": {
                "name": "memory_search",
                "arguments": arguments,
            },
        }
        try:
            resp = self._client.post(self._base_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                logger.warning("MemoryGate returned error: %s", data["error"])
                return []

            # MCP tool response: result.content[0].text is a JSON string
            content = data.get("result", {}).get("content", [])
            if not content:
                return []

            inner = json.loads(content[0].get("text", "{}"))
            return [
                MemoryResult(
                    text=r.get("text", ""),
                    source_type=r.get("source_type", "unknown"),
                    confidence=r.get("confidence", 0.0),
                )
                for r in inner.get("results", [])
            ]
        except Exception:
            logger.debug("MemoryGate search failed", exc_info=True)
            return []

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
