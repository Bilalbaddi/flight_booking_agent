"""
Configuration for the Flight Booking AI Agent.
Loads environment variables and exposes project-wide constants.
"""

import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# ── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"  # fast & capable model on Groq

# ── Target website ────────────────────────────────────────────────────────────
GOOGLE_FLIGHTS_URL: str = "https://www.google.com/travel/flights"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_PASSENGERS: int = 1
BROWSER_IMPLICIT_WAIT: int = 10  # seconds
