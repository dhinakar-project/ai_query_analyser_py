"""LLM utility module for Google Gemini integration."""

from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_gemini(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Get a Gemini LLM instance with the specified temperature.
    
    Args:
        temperature: The temperature for response generation.
                     Lower values (0.1) for deterministic outputs (classification/sentiment).
                     Higher values (0.7) for creative outputs (response generation).
    
    Returns:
        ChatGoogleGenerativeAI: Configured Gemini LLM instance.
    
    Raises:
        EnvironmentError: If GEMINI_API_KEY is not found in environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Please add your API key to the .env file.\n"
            "Example: GEMINI_API_KEY=your_api_key_here"
        )
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        convert_system_message_to_human=True
    )
