"""
Configuration settings for YouTube RAG Chatbot
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"  # Updated model
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2048

# Chunking Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Settings
TOP_K_RESULTS = 6
SIMILARITY_THRESHOLD = 0.3

# App Settings
APP_TITLE = "YouTube RAG Chatbot"
APP_ICON = "🎥"