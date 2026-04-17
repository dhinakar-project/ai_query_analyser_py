# AI-Based Customer Query Analyzer

An intelligent customer support chatbot built with **LangChain**, **Google Gemini**, and **Streamlit** that classifies customer queries, analyzes sentiment, and generates contextual, empathetic responses.

## Overview

This project implements a multi-agent pipeline where each agent is specialized for a specific task:

1. **Classifier Agent** - Categorizes queries into support categories
2. **Sentiment Agent** - Analyzes emotional tone
3. **Responder Agent** - Generates tailored support responses

## Features

- **Intelligent Classification**: Automatically categorizes queries into 6 categories (Billing, Technical Support, Returns & Refunds, Shipping & Delivery, Account Management, General Inquiry)
- **Sentiment Analysis**: Detects emotional states (Positive, Neutral, Negative, Urgent, Frustrated)
- **Empathetic Responses**: Generates context-aware, sentiment-matched customer support replies
- **Dark Theme UI**: Modern, professional interface with custom styling
- **Conversation History**: Track and review all analyzed queries

## Folder Structure

```
customer-query-analyzer/
├── .streamlit/
│   └── config.toml
├── agents/
│   ├── __init__.py
│   ├── classifier_agent.py
│   ├── sentiment_agent.py
│   └── responder_agent.py
├── tools/
│   ├── __init__.py
│   ├── classification_tool.py
│   ├── sentiment_tool.py
│   └── response_tool.py
├── utils/
│   ├── __init__.py
│   └── llm.py
├── app.py
├── requirements.txt
├── .env
└── README.md
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root and add your Google Gemini API key:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

You can obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   [Classifier Agent]                             │
│                   classify_query tool                            │
│                          │                                       │
│                          ▼                                       │
│                    Category Label                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    [Sentiment Agent]                             │
│                    analyze_sentiment tool                        │
│                          │                                       │
│                          ▼                                       │
│                     Sentiment Label                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    [Responder Agent]                             │
│                    generate_response tool                        │
│                          │                                       │
│                          ▼                                       │
│                    Final Response                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Streamlit UI displays:                               │
│           Category + Sentiment + Response + History               │
└─────────────────────────────────────────────────────────────────┘
```

## Query Categories

| Category | Description | Example |
|----------|-------------|---------|
| 💰 Billing | Payment and invoice issues | "I was charged twice for my order" |
| 🔧 Technical Support | Product/Service technical issues | "The app keeps crashing on startup" |
| 📦 Returns & Refunds | Product returns and refunds | "I want to return a damaged item" |
| 🚚 Shipping & Delivery | Order status and delivery | "Where is my order?" |
| 👤 Account Management | Account settings and access | "I can't reset my password" |
| 💬 General Inquiry | Other questions | "What are your business hours?" |

## Example Queries

Try these sample queries to test the analyzer:

- **Billing**: "My bill seems incorrect this month. I've been charged twice."
- **Technical Support**: "I can't login to my account, the password reset isn't working."
- **Shipping**: "Where is my order? It's been 2 weeks since I placed it!"
- **Returns**: "I received a damaged product and want to return it for a full refund."
- **Account**: "How do I upgrade my subscription to the premium plan?"
- **Positive**: "Your service has been amazing, just wanted to say thanks!"

## Technology Stack

- **LangChain**: Agent framework and tool orchestration
- **Google Gemini 2.0 Flash**: LLM for classification, sentiment analysis, and response generation
- **Streamlit**: User interface
- **Python**: Core programming language

## Architecture Highlights

### Temperature Settings
- **Classifier Agent**: 0.1 (deterministic, consistent classifications)
- **Sentiment Agent**: 0.1 (deterministic, consistent analysis)
- **Responder Agent**: 0.7 (creative, natural responses)

### Error Handling
All agents include robust error handling with fallback responses to ensure the application remains functional even if individual components fail.

## License

MIT License
