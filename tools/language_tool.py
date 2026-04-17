"""Language detection tool using Gemini LLM."""

import logging
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

from utils.llm import get_gemini

logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "ar": "Arabic",
    "ru": "Russian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "id": "Indonesian",
    "ms": "Malay",
    "cs": "Czech",
    "sk": "Slovak",
    "ro": "Romanian",
    "hu": "Hungarian",
    "uk": "Ukrainian"
}


def detect_language(query: str) -> Dict[str, str]:
    """Detect the language of a customer query.
    
    This tool uses the Gemini LLM to identify the language of the input text
    and returns both the language name and ISO 639-1 code.
    
    Args:
        query: The customer's query text to analyze.
        
    Returns:
        A dictionary containing:
        - language_name: Full name of the detected language (e.g., "English")
        - language_code: ISO 639-1 code (e.g., "en")
        
    Example:
        >>> result = detect_language.invoke("How do I reset my password?")
        >>> result
        {'language_name': 'English', 'language_code': 'en'}
    """
    logger.debug(f"Detecting language for query: {query[:50]}...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a language detection specialist. Your task is to identify the language of the given text.

Detect the primary language of the query and respond with EXACTLY ONE language name and its ISO 639-1 code.
Supported languages: {', '.join(f"{name} ({code})" for code, name in SUPPORTED_LANGUAGES.items())}

If the text contains multiple languages, detect the PRIMARY language (the one most used).

Respond in this EXACT format (no extra text):
LanguageName|LANGCODE

Examples:
- "Hello, how can I help you?" → English|en
- "¿Cómo puedo ayudarte?" → Spanish|es
- "Bonjour, comment allez-vous?" → French|fr
- "Ich habe ein Problem mit meiner Rechnung" → German|de
- "My billing address is wrong" → English|en
- "Quiero devolver mi pedido" → Spanish|es
- "Probleme mit Lieferung" → German|de
- "Come posso aiutarti?" → Italian|it
- "Wat is je probleem?" → Dutch|nl
- "J'ai un problème technique" → French|fr
- "Preciso de ajuda com minha conta" → Portuguese|pt"""),
        ("human", "Detect the language of this text: {query}")
    ])
    
    llm = get_gemini(temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"query": query})
        result = result.strip()
        
        if "|" in result:
            parts = result.split("|")
            if len(parts) >= 2:
                lang_name = parts[0].strip()
                lang_code = parts[1].strip().lower()
                
                if lang_code in SUPPORTED_LANGUAGES:
                    if lang_name.lower() != SUPPORTED_LANGUAGES[lang_code].lower():
                        lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_name)
                    
                    logger.debug(f"Detected language: {lang_name} ({lang_code})")
                    return {
                        "language_name": lang_name,
                        "language_code": lang_code
                    }
        
        logger.warning(f"Unexpected language detection format: {result}, defaulting to English")
        return {
            "language_name": "English",
            "language_code": "en"
        }
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return {
            "language_name": "English",
            "language_code": "en"
        }


def get_language_code(query: str) -> str:
    """Get just the language code for a query.
    
    Args:
        query: The customer's query text.
        
    Returns:
        ISO 639-1 language code string.
    """
    result = detect_language.invoke({"query": query})
    return result.get("language_code", "en")


def get_language_name(query: str) -> str:
    """Get just the language name for a query.
    
    Args:
        query: The customer's query text.
        
    Returns:
        Full language name string.
    """
    result = detect_language.invoke({"query": query})
    return result.get("language_name", "English")
