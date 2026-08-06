"""
aqi_utils.py
------------
Shared utility functions for working with AQI values and categories.

This module implements the official US EPA-style AQI breakpoints that are
used consistently by both the training pipeline and the Streamlit app, so
that a predicted numeric AQI value is always mapped to the same category,
color, and severity level everywhere in the system.
"""

# AQI breakpoints -> (label, color, severity_rank)
# severity_rank is used for sorting / progress bars (0 = best, 5 = worst)
AQI_BREAKPOINTS = [
    (0, 50, "Good", "#00A65A", 0),
    (50.01, 100, "Moderate", "#F4C542", 1),
    (100.01, 150, "Unhealthy for Sensitive Groups", "#FF8C42", 2),
    (150.01, 200, "Unhealthy", "#E9573F", 3),
    (200.01, 300, "Very Unhealthy", "#9C27B0", 4),
    (300.01, 1000, "Hazardous", "#6D0000", 5),
]


def value_to_category(aqi_value: float) -> str:
    """Convert a numeric AQI value into its official category label."""
    aqi_value = float(aqi_value)
    for low, high, label, _color, _rank in AQI_BREAKPOINTS:
        if low <= aqi_value <= high:
            return label
    # Anything above the highest breakpoint is still Hazardous
    if aqi_value > AQI_BREAKPOINTS[-1][1]:
        return "Hazardous"
    return "Good"


def category_color(category: str) -> str:
    """Return the hex color associated with an AQI category."""
    for _low, _high, label, color, _rank in AQI_BREAKPOINTS:
        if label == category:
            return color
    return "#808080"


def category_severity(category: str) -> int:
    """Return the severity rank (0-5) for a category, used for progress bars."""
    for _low, _high, label, _color, rank in AQI_BREAKPOINTS:
        if label == category:
            return rank
    return 0


def category_from_and_color(aqi_value: float):
    """Convenience helper returning (category, color, severity) in one call."""
    category = value_to_category(aqi_value)
    return category, category_color(category), category_severity(category)
