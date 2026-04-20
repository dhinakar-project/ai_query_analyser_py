"""Vapi assistant configuration builder."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

VOICE_SYSTEM_PROMPT = """You are an empathetic AI customer support voice agent.
When the customer describes their issue:
1. Listen carefully and ask one clarifying question if needed
2. Identify the category: Billing, Technical Support, Returns & Refunds,
   Shipping & Delivery, Account Management, or General Inquiry
3. Acknowledge their sentiment appropriately
4. Provide a helpful response and clear next steps
Keep responses brief (2-3 sentences max) since this is a voice call."""


def build_customer_support_assistant_config(
    name: str = "AI Query Analyzer Voice Agent",
    model_provider: str = "google",
    model: str = "gemini-2.0-flash",
    temperature: float = 0.4,
    voice_provider: str = "vapi",
    voice_id: str = "听到",
    transcriber_provider: str = "deepgram",
    transcriber_model: str = "nova-2",
    first_message: str = "Hello! I'm your AI support agent. How can I help you today?",
) -> Dict[str, Any]:
    """Build Vapi assistant configuration for customer support.

    Args:
        name: Assistant name
        model_provider: LLM provider (google, openai, etc.)
        model: Model name
        temperature: LLM temperature
        voice_provider: TTS provider (11labs, deepgram, openai)
        voice_id: Voice ID to use
        transcriber_provider: STT provider
        transcriber_model: STT model
        first_message: First message when call starts

    Returns:
        Vapi assistant configuration dict
    """
    config = {
        "name": name,
        "model": {
            "provider": model_provider,
            "model": model,
            "systemPrompt": VOICE_SYSTEM_PROMPT,
            "temperature": temperature,
        },
        "voice": {
            "provider": voice_provider,
            "voiceId": voice_id,
        },
        "transcriber": {
            "provider": transcriber_provider,
            "model": transcriber_model,
            "language": "en-US",
        },
        "firstMessage": first_message,
        "recordingEnabled": True,
        "endCallFunctionEnabled": True,
    }

    logger.debug(f"Built assistant config: {name}")
    return config