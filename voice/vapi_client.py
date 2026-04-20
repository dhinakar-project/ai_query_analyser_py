"""Vapi REST API client for voice agent integration."""

import logging
import os
from typing import Optional, Dict, Any, List

import httpx

logger = logging.getLogger(__name__)


class VapiError(Exception):
    """Custom exception for Vapi API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class VapiClient:
    """Vapi REST API client."""

    BASE_URL = "https://api.vapi.ai"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Vapi client.

        Args:
            api_key: Vapi API key. If None, reads from VAPI_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("VAPI_API_KEY")
        if not self.api_key:
            raise VapiError("VAPI_API_KEY not found. Set it in .env or pass as parameter.")
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    def _request_with_retry(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            url: Full URL to request
            **kwargs: Additional arguments to pass to httpx

        Returns:
            Parsed JSON response

        Raises:
            VapiError: On API errors
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self._client.request(method, url, **kwargs)

                if response.status_code == 429:
                    retry_count += 1
                    if retry_count < max_retries:
                        import time
                        wait_time = 2 ** retry_count
                        logger.warning(f"Rate limited, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise VapiError(
                        "Rate limit exceeded after retries",
                        status_code=429
                    )

                if response.status_code >= 400:
                    error_body = response.json() if response.content else {}
                    raise VapiError(
                        error_body.get("message", "API error"),
                        status_code=response.status_code,
                        response=error_body
                    )

                return response.json()

            except httpx.RequestError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise VapiError(f"Request failed: {str(e)}")
                import time
                time.sleep(2 ** retry_count)

        raise VapiError("Max retries exceeded")

    def create_assistant(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Vapi assistant.

        Args:
            config: Assistant configuration dict

        Returns:
            Assistant configuration dict with ID
        """
        return self._request_with_retry("POST", "/assistant", json=config)

    def get_assistant(self, assistant_id: str) -> Dict[str, Any]:
        """Get assistant by ID.

        Args:
            assistant_id: Assistant ID

        Returns:
            Assistant configuration dict
        """
        return self._request_with_retry("GET", f"/assistant/{assistant_id}")

    def list_assistants(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List assistants.

        Args:
            limit: Maximum number of assistants to return

        Returns:
            List of assistant dicts
        """
        result = self._request_with_retry("GET", "/assistant", params={"limit": limit})
        if isinstance(result, list):
            return result
        return result.get("assistants", []) if isinstance(result, dict) else []

    def delete_assistant(self, assistant_id: str) -> Dict[str, Any]:
        """Delete an assistant.

        Args:
            assistant_id: Assistant ID to delete

        Returns:
            Deletion confirmation
        """
        return self._request_with_retry("DELETE", f"/assistant/{assistant_id}")

    def start_phone_call(
        self,
        assistant_id: str,
        phone_number: str,
        caller_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start an outbound phone call.

        Args:
            assistant_id: Assistant ID to use
            phone_number: Phone number to call (E.164 format, e.g., +1234567890)
            caller_id: Optional caller ID to display

        Returns:
            Call configuration dict with call ID
        """
        payload = {
            "assistantId": assistant_id,
            "phoneNumber": phone_number,
        }
        if caller_id:
            payload["callerId"] = caller_id

        return self._request_with_retry("POST", "/call/phone", json=payload)

    def start_web_call(self, assistant_id: str, public_key: str = None) -> Dict[str, Any]:
        """Start an in-browser web call.

        Args:
            assistant_id: Assistant ID to use
            public_key: Vapi public key (from VAPI_PUBLIC_KEY env var)

        Returns:
            Call configuration dict with join URL
        """
        public_key = public_key or os.getenv("VAPI_PUBLIC_KEY")
        if not public_key:
            raise VapiError("VAPI_PUBLIC_KEY not found. Add it to your .env file.")

        response = httpx.post(
            f"{self.BASE_URL}/call/web",
            headers={
                "Authorization": f"Bearer {public_key}",
                "Content-Type": "application/json",
            },
            json={"assistantId": assistant_id},
            timeout=30.0,
        )
        if response.status_code >= 400:
            error_body = response.json() if response.content else {}
            raise VapiError(
                error_body.get("message", "API error"),
                status_code=response.status_code,
                response=error_body
            )
        return response.json()

    def get_call(self, call_id: str) -> Dict[str, Any]:
        """Get call details.

        Args:
            call_id: Call ID

        Returns:
            Call configuration dict
        """
        return self._request_with_retry("GET", f"/call/{call_id}")

    def list_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of call dicts
        """
        result = self._request_with_retry("GET", "/call", params={"limit": limit})
        if isinstance(result, list):
            return result
        return result.get("calls", []) if isinstance(result, dict) else []

    def end_call(self, call_id: str) -> Dict[str, Any]:
        """End an active call.

        Args:
            call_id: Call ID to end

        Returns:
            Call configuration dict
        """
        return self._request_with_retry("DELETE", f"/call/{call_id}")

    def get_call_transcript(self, call_id: str) -> str:
        """Get transcript text from a call.

        Args:
            call_id: Call ID

        Returns:
            Transcript text
        """
        call_data = self.get_call(call_id)
        artifacts = call_data.get("artifact", {})

        transcript_parts = []
        for message in artifacts.get("messages", []):
            if message.get("role") == "customer":
                transcript_parts.append(f"Customer: {message.get('text', '')}")
            elif message.get("role") == "agent":
                transcript_parts.append(f"Agent: {message.get('text', '')}")

        return "\n".join(transcript_parts) if transcript_parts else artifacts.get("transcript", "")

    def get_call_status(self, call_id: str) -> str:
        """Get call status.

        Args:
            call_id: Call ID

        Returns:
            Status string (in-progress, completed, ended)
        """
        call_data = self.get_call(call_id)
        return call_data.get("status", "unknown")

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()