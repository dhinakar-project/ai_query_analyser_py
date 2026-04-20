"""FastAPI webhook server for Vapi voice events."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app_state: Dict[str, Any] = {}


class VapiWebhookEvent(BaseModel):
    """Vapi webhook event payload."""

    event: str
    call_id: Optional[str] = None
    timestamp: Optional[str] = None
    message: Optional[Dict[str, Any]] = None


class TranscriptMessage(BaseModel):
    """Transcript message from Vapi."""

    role: str
    text: str


async def process_call_ended(call_id: str, payload: Dict[str, Any]) -> None:
    """Process call-ended event.

    Extracts transcript and runs through LangGraph pipeline.

    Args:
        call_id: Call ID
        payload: Full webhook payload
    """
    try:
        artifact = payload.get("artifact", {})
        messages = artifact.get("messages", [])

        transcript_parts = []
        for msg in messages:
            if msg.get("role") == "customer":
                transcript_parts.append(f"Customer: {msg.get('text', '')}")
            elif msg.get("role") == "agent":
                transcript_parts.append(f"Agent: {msg.get('text', '')}")

        transcript = "\n".join(transcript_parts)

        if transcript:
            logger.info(f"Processing transcript for call {call_id}")

            try:
                from voice.transcript_processor import process_voice_transcript
                result = await process_voice_transcript(transcript, call_id)
                logger.info(f"Analysis complete for call {call_id}: {result.get('category')} / {result.get('sentiment')}")

                app_state[f"call_{call_id}_result"] = result
            except Exception as e:
                logger.error(f"Failed to process transcript: {e}")
                app_state[f"call_{call_id}_result"] = {"error": str(e)}
        else:
            logger.warning(f"No transcript found for call {call_id}")

        app_state[f"call_{call_id}_status"] = "ended"

    except Exception as e:
        logger.error(f"Error processing call-ended: {e}")


async def process_transcript_event(call_id: str, payload: Dict[str, Any]) -> None:
    """Process transcript event (partial or final).

    Args:
        call_id: Call ID
        payload: Full webhook payload
    """
    try:
        message = payload.get("message", {})
        transcript_text = message.get("text", "")
        role = message.get("role", "unknown")

        if transcript_text:
            existing = app_state.get(f"call_{call_id}_transcript", "")
            app_state[f"call_{call_id}_transcript"] = f"{existing}\n{role}: {transcript_text}"
            logger.debug(f"Transcript update for {call_id}: {transcript_text[:50]}...")

    except Exception as e:
        logger.error(f"Error processing transcript: {e}")


@app.post("/webhook/vapi")
async def vapi_webhook(request: Request) -> Dict[str, str]:
    """Handle Vapi webhook events.

    Args:
        request: FastAPI request

    Returns:
        Success response
    """
    try:
        payload = await request.json()
        event_type = payload.get("event")
        call_id = payload.get("call_id")

        logger.info(f"Vapi webhook event: {event_type}, call_id: {call_id}")

        if not call_id:
            return {"status": "ignored", "reason": "no call_id"}

        if event_type == "call-started":
            app_state[f"call_{call_id}_status"] = "in-progress"
            logger.info(f"Call started: {call_id}")

        elif event_type == "transcript":
            await process_transcript_event(call_id, payload)

        elif event_type == "call-ended":
            await process_call_ended(call_id, payload)
            logger.info(f"Call ended: {call_id}")

        elif event_type == "function-call":
            logger.debug(f"Function call: {payload.get('function', {}).get('name')}")

        else:
            logger.debug(f"Unhandled event type: {event_type}")

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/webhook/status/{call_id}")
async def get_call_status(call_id: str) -> Dict[str, Any]:
    """Get status of a call.

    Args:
        call_id: Call ID

    Returns:
        Status dict with transcript and result
    """
    status = app_state.get(f"call_{call_id}_status", "unknown")
    transcript = app_state.get(f"call_{call_id}_transcript", "")
    result = app_state.get(f"call_{call_id}_result", None)

    return {
        "call_id": call_id,
        "status": status,
        "transcript": transcript,
        "result": result,
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Vapi webhook server started")
    yield
    logger.info("Vapi webhook server stopped")


def run_webhook_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the webhook server.

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    import uvicorn
    app = FastAPI(title="Vapi Webhook Server", lifespan=lifespan)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_webhook_server()