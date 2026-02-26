"""FastAPI application for the PCDC chat API server."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from pcdc.server.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaMessage,
    PCDCMetadata,
    UsageInfo,
)

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    host: str = "0.0.0.0"
    port: int = 8000
    pc_hidden_sizes: list[int] | None = None
    pc_alpha: float = 0.5
    pc_energy_scale: float = 1.0
    pc_checkpoint: str | None = None


# Llama-3 chat template
LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_HEADER = "<|start_header_id|>{role}<|end_header_id|>\n\n"
LLAMA3_EOT = "<|eot_id|>"


def format_llama3_prompt(messages: list[ChatMessage]) -> str:
    """Format messages using the Llama-3 chat template."""
    parts = [LLAMA3_BOS]
    for msg in messages:
        parts.append(LLAMA3_HEADER.format(role=msg.role))
        parts.append(msg.content)
        parts.append(LLAMA3_EOT)
    # Add assistant header to prime generation
    parts.append(LLAMA3_HEADER.format(role="assistant"))
    return "".join(parts)


def create_app(config: ServerConfig) -> FastAPI:
    """Factory that creates the FastAPI app with lifespan."""

    # Will be populated at startup
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # --- Startup: load model + PCHead ---
        logger.info("Loading GGUF model from %s ...", config.model_path)

        from pcdc.gguf.gguf_backend import GGUFBackend
        from pcdc.gguf.pc_head import PCHead
        from pcdc.server.steering import SteeringEngine

        backend = GGUFBackend(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            n_gpu_layers=config.n_gpu_layers,
        )
        hidden_dim = backend.hidden_dim
        logger.info("Model loaded: hidden_dim=%d", hidden_dim)

        hidden_sizes = config.pc_hidden_sizes or [1024, 256]
        pc_head = PCHead(
            feature_dim=hidden_dim,
            num_classes=hidden_dim,
            hidden_sizes=hidden_sizes,
        )
        pc_head.eval()
        logger.info("PCHead initialized: [%d, %s, %d]", hidden_dim, hidden_sizes, hidden_dim)

        engine = SteeringEngine(
            backend=backend,
            pc_head=pc_head,
            alpha=config.pc_alpha,
            energy_scale=config.pc_energy_scale,
        )

        # Load checkpoint if provided
        if config.pc_checkpoint:
            from pathlib import Path
            if Path(config.pc_checkpoint).exists():
                engine.load_checkpoint(config.pc_checkpoint)
            else:
                logger.info("No checkpoint at %s — starting fresh", config.pc_checkpoint)

        state["backend"] = backend
        state["engine"] = engine
        state["config"] = config

        # atexit fallback — Windows doesn't reliably fire lifespan shutdown
        if config.pc_checkpoint:
            def _save_on_exit():
                try:
                    engine.save_checkpoint(config.pc_checkpoint)
                except Exception:
                    pass
            atexit.register(_save_on_exit)

        logger.info("PCDC server ready on %s:%d", config.host, config.port)
        yield

        # --- Shutdown: save checkpoint ---
        if config.pc_checkpoint:
            engine.save_checkpoint(config.pc_checkpoint)
        logger.info("Shutting down PCDC server")

    app = FastAPI(title="PCDC Chat API", lifespan=lifespan)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "pcdc",
                    "object": "model",
                    "owned_by": "pcdc",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        engine = state["engine"]
        backend = state["backend"]

        prompt = format_llama3_prompt(request.messages)

        # Steer: embed + PCHead settle → adjusted temperature
        steering = await asyncio.to_thread(engine.steer, prompt, request.temperature)

        pcdc_meta = PCDCMetadata(
            settling_energy=steering.energy,
            converged=steering.converged,
            adjusted_temperature=steering.adjusted_temp,
            settle_steps=steering.settle_steps,
        )

        # Prepare stop sequences
        stop = request.stop
        if isinstance(stop, str):
            stop = [stop]

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if request.stream:
            return EventSourceResponse(
                _stream_completion(
                    backend, prompt, steering, pcdc_meta,
                    completion_id, request, stop,
                ),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming
            result = await asyncio.to_thread(
                backend.llm.create_chat_completion,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                temperature=steering.adjusted_temp,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=stop or [],
            )

            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})

            # Background: buffer for replay
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(asyncio.to_thread(engine.record, content))
            )

            return ChatCompletionResponse(
                id=completion_id,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(role="assistant", content=content),
                        finish_reason=result["choices"][0].get("finish_reason", "stop"),
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
                pcdc=pcdc_meta,
            )

    async def _stream_completion(
        backend,
        prompt: str,
        steering,
        pcdc_meta: PCDCMetadata,
        completion_id: str,
        request: ChatCompletionRequest,
        stop: list[str] | None,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE events in OpenAI streaming format."""

        # First chunk: role
        first_chunk = ChatCompletionChunk(
            id=completion_id,
            choices=[
                ChatCompletionChunkChoice(
                    delta=DeltaMessage(role="assistant"),
                )
            ],
            pcdc=pcdc_meta,
        )
        yield json.dumps(first_chunk.model_dump(), ensure_ascii=False)

        # Stream tokens
        def _generate():
            return backend.llm.create_chat_completion(
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                temperature=steering.adjusted_temp,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=stop or [],
                stream=True,
            )

        stream = await asyncio.to_thread(_generate)

        collected_text = []
        for chunk_data in stream:
            delta = chunk_data["choices"][0].get("delta", {})
            content = delta.get("content")
            finish_reason = chunk_data["choices"][0].get("finish_reason")

            if content:
                collected_text.append(content)
                sse_chunk = ChatCompletionChunk(
                    id=completion_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=DeltaMessage(content=content),
                        )
                    ],
                )
                yield json.dumps(sse_chunk.model_dump(), ensure_ascii=False)

            if finish_reason:
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=DeltaMessage(),
                            finish_reason=finish_reason,
                        )
                    ],
                )
                yield json.dumps(final_chunk.model_dump(), ensure_ascii=False)

        yield "[DONE]"

        # Background: buffer for replay
        full_text = "".join(collected_text)
        if full_text:
            engine = state["engine"]
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(asyncio.to_thread(engine.record, full_text))
            )

    @app.get("/v1/pcdc/stats")
    async def pcdc_stats():
        engine = state["engine"]
        return engine.get_stats()

    @app.post("/v1/pcdc/checkpoint")
    async def pcdc_checkpoint():
        cfg = state["config"]
        if not cfg.pc_checkpoint:
            return JSONResponse(
                status_code=400,
                content={"error": "no_checkpoint_path", "message": "Server started without --pc-checkpoint."},
            )
        engine = state["engine"]
        await asyncio.to_thread(engine.save_checkpoint, cfg.pc_checkpoint)
        return {"status": "saved", "path": cfg.pc_checkpoint}

    @app.post("/v1/pcdc/train")
    async def pcdc_train():
        return JSONResponse(
            status_code=501,
            content={"error": "not_implemented_v1", "message": "Online training not yet available."},
        )

    return app


def main():
    """CLI entry point for pcdc-serve."""
    parser = argparse.ArgumentParser(description="PCDC Chat API Server")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context window size")
    parser.add_argument("--n-threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU offload layers")
    parser.add_argument("--pc-alpha", type=float, default=0.5, help="Energy-temperature coupling strength")
    parser.add_argument("--pc-energy-scale", type=float, default=1.0, help="Energy normalization scale")
    parser.add_argument("--pc-checkpoint", default=None, help="Path to save/load PCHead checkpoint (auto-saves on shutdown)")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = ServerConfig(
        model_path=args.model,
        host=args.host,
        port=args.port,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        pc_alpha=args.pc_alpha,
        pc_energy_scale=args.pc_energy_scale,
        pc_checkpoint=args.pc_checkpoint,
    )

    import uvicorn
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
