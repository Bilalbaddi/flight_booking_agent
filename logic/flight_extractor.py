"""
Flight results extraction logic for Google Flights.

This module waits for the search results page to load and extracts structured
flight details from visible result cards.
"""

import re
from typing import List, Dict

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException


_PRICE_RE = re.compile(r"(?:₹|Rs\.?|INR)\s?[\d,]+", re.IGNORECASE)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)\b")
_DURATION_RE = re.compile(
    r"\b\d+\s*(?:h|hr|hrs|hour|hours)\s*\d*\s*(?:m|min|mins|minute|minutes)?\b",
    re.IGNORECASE,
)
_STOPS_RE = re.compile(r"\bnonstop\b|\b\d+\s*stop(?:s)?\b", re.IGNORECASE)

_AIRLINE_HINTS = [
    "IndiGo", "Air India", "Vistara", "Akasa Air", "SpiceJet", "AirAsia",
    "Qatar Airways", "Emirates", "Etihad", "Lufthansa", "British Airways",
    "Singapore Airlines", "Thai Airways", "Malaysia Airlines", "KLM", "Air France",
]


def _normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\u202f", " ").replace("\xa0", " ").split())


def _pick_airline(text: str) -> str:
    # 1) Explicit "Operated by X"
    m = re.search(r"operated by\s+([A-Za-z0-9 .&-]+)", text, re.IGNORECASE)
    if m:
        return _normalize_text(m.group(1).strip(" .,"))

    # 2) Known airline names in text
    lower = text.lower()
    for name in _AIRLINE_HINTS:
        if name.lower() in lower:
            return name

    # 3) Fallback: best-effort first phrase before separator
    parts = re.split(r"\.|•|\||,", text)
    for part in parts:
        chunk = _normalize_text(part)
        if 3 <= len(chunk) <= 40 and "flight" not in chunk.lower() and "from" not in chunk.lower():
            return chunk

    return ""


def _extract_from_text(text: str) -> Dict[str, str]:
    t = _normalize_text(text)

    price_match = _PRICE_RE.search(t)
    times = _TIME_RE.findall(t)
    duration_match = _DURATION_RE.search(t)
    stops_match = _STOPS_RE.search(t)

    departure = times[0] if len(times) >= 1 else ""
    arrival = times[1] if len(times) >= 2 else ""

    return {
        "airline": _pick_airline(t),
        "departure": departure,
        "arrival": arrival,
        "departure_time": departure,
        "arrival_time": arrival,
        "duration": duration_match.group(0) if duration_match else "",
        "stops": stops_match.group(0).lower() if stops_match else "",
        "price": price_match.group(0) if price_match else "",
    }


def _is_result_like(item: Dict[str, str]) -> bool:
    # Require at least two strong fields to avoid noise cards.
    strong = 0
    if item.get("airline"):
        strong += 1
    if item.get("departure") and item.get("arrival"):
        strong += 1
    if item.get("price"):
        strong += 1
    if item.get("duration"):
        strong += 1
    return strong >= 2


def _collect_candidate_texts(driver, limit: int = 80) -> List[str]:
    """
    Collect visible card text blocks using structural XPath and generic content
    signals (₹, CO2e, time patterns). Avoids obfuscated CSS classes.
    """
    candidates = []

    xpaths = [
        # Primary Google Flights list structure.
        "//ul[@role='listbox']/li",
        # Generic listitem containers.
        "//*[@role='listitem']",
        # Generic blocks that contain a price indicator.
        "//div[contains(normalize-space(.), '₹') or contains(normalize-space(.), 'CO2e')]",
    ]

    seen_texts = set()
    for xp in xpaths:
        try:
            nodes = driver.find_elements(By.XPATH, xp)
        except Exception:
            continue

        for node in nodes:
            try:
                if not node.is_displayed():
                    continue
                raw = _normalize_text(node.text)
                if not raw or len(raw) < 20:
                    continue

                has_price = _PRICE_RE.search(raw) is not None or "₹" in raw
                has_co2 = "co2e" in raw.lower()
                has_time = _TIME_RE.search(raw) is not None
                has_stops = _STOPS_RE.search(raw) is not None

                if has_price and (has_time or has_stops or has_co2):
                    if raw not in seen_texts:
                        seen_texts.add(raw)
                        candidates.append(raw)
                        if len(candidates) >= limit:
                            return candidates
            except Exception:
                continue

    return candidates


def wait_for_results_container(driver, timeout: int = 30) -> None:
    """
    Dynamic wait until results are present.
    Uses stable text indicators (₹ or CO2e) plus card discovery.
    """
    wait = WebDriverWait(driver, timeout, poll_frequency=0.5)

    def indicators_visible(d):
        try:
            nodes = d.find_elements(
                By.XPATH,
                "//*[contains(normalize-space(.), '₹') or contains(translate(normalize-space(.), 'COE', 'coe'), 'co2e')]"
            )
            return any(n.is_displayed() for n in nodes)
        except Exception:
            return False

    try:
        wait.until(indicators_visible)
        wait.until(lambda d: len(_collect_candidate_texts(d, limit=30)) > 0)
    except TimeoutException as exc:
        raise RuntimeError(
            "Flight results not ready: no visible price/CO2e indicator or flight cards found within timeout."
        ) from exc


def extract_flight_results(driver, max_results: int = 10, timeout: int = 30) -> List[Dict[str, str]]:
    """
    Extract structured flights from Google Flights results page.

    Returns a list in shape:
    [
      {
        "airline": "",
        "departure": "",
        "arrival": "",
        "duration": "",
        "stops": "",
        "price": ""
      }
    ]
    """
    wait_for_results_container(driver, timeout=timeout)

    raw_texts = _collect_candidate_texts(driver, limit=max(60, max_results * 8))

    flights: List[Dict[str, str]] = []
    seen_keys = set()

    for text in raw_texts:
        item = _extract_from_text(text)
        if not _is_result_like(item):
            continue

        key = (
            item.get("airline", ""),
            item.get("departure", ""),
            item.get("arrival", ""),
            item.get("price", ""),
        )
        if key in seen_keys:
            continue

        seen_keys.add(key)
        flights.append(item)
        if len(flights) >= max_results:
            break

    if not flights:
        raise RuntimeError("Flight cards were found, but parsing produced 0 structured results.")

    return flights
