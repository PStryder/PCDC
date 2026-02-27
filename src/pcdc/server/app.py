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
    pc_beta: float = 0.5
    pc_online_eta_w: float = 0.0001
    pc_replay_k: int = 4
    pc_settle_k: int | None = None
    # MemoryGate retrieval
    mg_url: str | None = None
    mg_timeout: float = 2.0
    mg_bearer_token: str | None = None
    mg_recon_threshold: float = float("inf")
    mg_predict_threshold: float = float("inf")
    mg_predict_percentile: float | None = None
    mg_retrieval_limit: int = 3
    mg_retrieval_min_confidence: float = 0.5
    # Deviation routing
    pc_deviation_routing: bool = False
    pc_deviation_threshold: float = 0.6
    # Telemetry
    telemetry_db: str | None = None


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


def format_augmented_prompt(messages: list[dict], memory_results: list) -> str:
    """Rebuild Llama-3 prompt with retrieved memory context injected.

    Inserts a system message at position 0 containing the retrieved
    memory items, then re-formats the full message list.
    """
    context_lines = []
    for i, r in enumerate(memory_results, 1):
        context_lines.append(f"[{i}] ({r.source_type}, conf={r.confidence:.2f}) {r.text}")
    context_block = "\n".join(context_lines)

    context_msg = ChatMessage(
        role="system",
        content=f"Relevant context from memory:\n{context_block}",
    )
    original_msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
    return format_llama3_prompt([context_msg] + original_msgs)


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

        # Memory client (optional — only created when --mg-url is set)
        memory_client = None
        if config.mg_url:
            from pcdc.server.memory_client import MemoryClient
            memory_client = MemoryClient(
                base_url=config.mg_url,
                timeout=config.mg_timeout,
                bearer_token=config.mg_bearer_token,
            )
            logger.info("MemoryGate client initialized: %s", config.mg_url)

        # Telemetry DB (optional — only created when --telemetry-db is set)
        telemetry = None
        if config.telemetry_db:
            from pcdc.server.telemetry_db import TelemetryDB
            telemetry = TelemetryDB(config.telemetry_db)
            logger.info("TelemetryDB initialized: %s", config.telemetry_db)

        engine = SteeringEngine(
            backend=backend,
            pc_head=pc_head,
            alpha=config.pc_alpha,
            energy_scale=config.pc_energy_scale,
            beta=config.pc_beta,
            online_eta_w=config.pc_online_eta_w,
            replay_k=config.pc_replay_k,
            settle_k=config.pc_settle_k,
            memory_client=memory_client,
            retrieval_recon_threshold=config.mg_recon_threshold,
            retrieval_predict_threshold=config.mg_predict_threshold,
            retrieval_predict_percentile=config.mg_predict_percentile,
            retrieval_limit=config.mg_retrieval_limit,
            retrieval_min_confidence=config.mg_retrieval_min_confidence,
            format_prompt_fn=format_augmented_prompt if memory_client else None,
            deviation_routing_enabled=config.pc_deviation_routing,
            deviation_routing_threshold=config.pc_deviation_threshold,
            telemetry_db=telemetry,
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

        # --- Shutdown: save checkpoint, close clients ---
        if config.pc_checkpoint:
            engine.save_checkpoint(config.pc_checkpoint)
        if memory_client:
            memory_client.close()
        if telemetry:
            telemetry.close()
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

        prompt = format_llama3_prompt(request.messages)

        # Prepare stop sequences
        stop = request.stop
        if isinstance(stop, str):
            stop = [stop]

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if request.stream:
            messages_raw = [{"role": m.role, "content": m.content} for m in request.messages]
            return EventSourceResponse(
                _stream_completion(
                    engine, prompt, completion_id, request, stop, messages_raw,
                ),
                media_type="text/event-stream",
            )
        else:
            # Raw messages for potential retrieval-triggered prompt rebuild
            messages_raw = [{"role": m.role, "content": m.content} for m in request.messages]

            # Single atomic call: embed → steer → retrieval gate → generate
            steering, result = await asyncio.to_thread(
                engine.steer_and_generate,
                prompt,
                request.temperature,
                messages=messages_raw,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=stop or [],
            )

            pcdc_meta = PCDCMetadata(
                settling_energy=steering.energy,
                reconstruction_energy=steering.reconstruction_energy,
                predictive_energy=steering.predictive_energy,
                converged=steering.converged,
                adjusted_temperature=steering.adjusted_temp,
                settle_steps=steering.settle_steps,
                cosine_distance=steering.cosine_distance,
                retrieval_triggered=steering.retrieval_triggered,
                retrieval_count=steering.retrieval_count,
                deviation_match_score=steering.deviation_match_score,
            )

            content = result["choices"][0]["text"]
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
        engine,
        prompt: str,
        completion_id: str,
        request: ChatCompletionRequest,
        stop: list[str] | None,
        messages_raw: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE events in OpenAI streaming format."""

        # Single atomic call: embed → steer → retrieval gate → generate
        def _start():
            return engine.steer_and_generate(
                prompt,
                request.temperature,
                messages=messages_raw,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=stop or [],
                stream=True,
            )

        steering, stream = await asyncio.to_thread(_start)

        pcdc_meta = PCDCMetadata(
            settling_energy=steering.energy,
            reconstruction_energy=steering.reconstruction_energy,
            predictive_energy=steering.predictive_energy,
            converged=steering.converged,
            adjusted_temperature=steering.adjusted_temp,
            settle_steps=steering.settle_steps,
            cosine_distance=steering.cosine_distance,
            retrieval_triggered=steering.retrieval_triggered,
            retrieval_count=steering.retrieval_count,
            deviation_match_score=steering.deviation_match_score,
        )

        # First chunk: role + pcdc metadata
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

        # Stream tokens (iterator holds the steering lock until exhausted)
        collected_text = []
        for chunk_data in stream:
            content = chunk_data["choices"][0].get("text")
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
    parser.add_argument("--pc-beta", type=float, default=0.5, help="Blend ratio: energy = beta*E_recon + (1-beta)*E_predict (default: 0.5)")
    parser.add_argument("--pc-online-eta-w", type=float, default=0.0001, help="Online learning rate for train_step phases (default: 0.0001)")
    parser.add_argument("--pc-replay-k", type=int, default=4, help="Replay samples per training phase (default: 4)")
    parser.add_argument("--pc-settle-k", type=int, default=None, help="Override settle steps per phase (default: PCHead config K)")
    # MemoryGate retrieval
    parser.add_argument("--mg-url", default=None, help="MemoryGate MCP endpoint URL (enables retrieval)")
    parser.add_argument("--mg-timeout", type=float, default=2.0, help="MemoryGate request timeout in seconds (default: 2.0)")
    parser.add_argument("--mg-bearer-token", default=None, help="Bearer token for MemoryGate auth")
    parser.add_argument("--mg-recon-threshold", type=float, default=float("inf"), help="Reconstruction energy threshold for retrieval (default: disabled)")
    parser.add_argument("--mg-predict-threshold", type=float, default=float("inf"), help="Predictive energy threshold for retrieval (default: disabled)")
    parser.add_argument("--mg-predict-percentile", type=float, default=None, help="Dynamic retrieval threshold as percentile of predict energy history (e.g., 95)")
    parser.add_argument("--mg-retrieval-limit", type=int, default=3, help="Max memory items to retrieve (default: 3)")
    parser.add_argument("--mg-retrieval-min-confidence", type=float, default=0.5, help="Min confidence for retrieved memories (default: 0.5)")
    # Deviation routing
    parser.add_argument("--pc-deviation-routing", action="store_true", default=False, help="Enable deviation-based retrieval routing")
    parser.add_argument("--pc-deviation-threshold", type=float, default=0.6, help="Cosine similarity threshold for deviation routing (default: 0.6)")
    # Telemetry
    parser.add_argument("--telemetry-db", default=None, help="Path to SQLite telemetry database (default: disabled)")
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
        pc_beta=args.pc_beta,
        pc_online_eta_w=args.pc_online_eta_w,
        pc_replay_k=args.pc_replay_k,
        pc_settle_k=args.pc_settle_k,
        mg_url=args.mg_url,
        mg_timeout=args.mg_timeout,
        mg_bearer_token=args.mg_bearer_token,
        mg_recon_threshold=args.mg_recon_threshold,
        mg_predict_threshold=args.mg_predict_threshold,
        mg_predict_percentile=args.mg_predict_percentile,
        mg_retrieval_limit=args.mg_retrieval_limit,
        mg_retrieval_min_confidence=args.mg_retrieval_min_confidence,
        pc_deviation_routing=args.pc_deviation_routing,
        pc_deviation_threshold=args.pc_deviation_threshold,
        telemetry_db=args.telemetry_db,
    )

    import uvicorn
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
