"""
Price comparison utilities for extracted flight results.

Provides helpers to:
- parse currency price text (e.g., "₹11,699") into integers
- sort flights by price (lowest to highest)
- identify the cheapest flight
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def parse_price_to_int(price_text: str) -> Optional[int]:
    """
    Convert a price string into an integer.

    Examples:
        "₹11,699" -> 11699
        "INR 9,250" -> 9250

    Returns None if no numeric value is found.
    """
    if not price_text:
        return None

    digits = re.sub(r"[^\d]", "", str(price_text))
    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def sort_flights_by_price(flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return a new list of flights sorted from lowest price to highest.

    Flights with invalid/missing prices are placed at the end.
    """
    enriched: List[Dict[str, Any]] = []
    for flight in flights or []:
        item = dict(flight)
        item["price_value"] = parse_price_to_int(item.get("price", ""))
        enriched.append(item)

    # Put missing prices at the end by using a very large fallback key.
    sorted_list = sorted(
        enriched,
        key=lambda f: f["price_value"] if f.get("price_value") is not None else 10**12,
    )
    return sorted_list


def find_cheapest_flight(
    flights: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Return both:
    1) sorted flight list (low -> high)
    2) cheapest flight object (or None if not available)
    """
    sorted_flights = sort_flights_by_price(flights)

    cheapest = None
    for flight in sorted_flights:
        if flight.get("price_value") is not None:
            cheapest = flight
            break

    return sorted_flights, cheapest


def format_cheapest_flight_report(cheapest_flight: Optional[Dict[str, Any]]) -> str:
    """
    Build a user-friendly summary block for the cheapest flight.
    """
    if not cheapest_flight:
        return "Cheapest Flight Found\nNo valid priced flights were found."

    return (
        "Cheapest Flight Found\n"
        f"Airline: {cheapest_flight.get('airline', '')}\n"
        f"Departure: {cheapest_flight.get('departure', '')}\n"
        f"Arrival: {cheapest_flight.get('arrival', '')}\n"
        f"Duration: {cheapest_flight.get('duration', '')}\n"
        f"Stops: {cheapest_flight.get('stops', '')}\n"
        f"Price: {cheapest_flight.get('price', '')}"
    )
