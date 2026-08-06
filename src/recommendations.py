"""
recommendations.py
-------------------
A standalone, reusable module that maps an AQI category to:
  - a human-readable health risk description
  - a list of personal recommendations
  - a list of preventive measures
  - the group most affected

Kept separate from the Streamlit UI code and the ML code so it can be
imported, tested, and explained independently (good practice to highlight
in a dissertation viva: separation of concerns).
"""

RECOMMENDATIONS = {
    "Good": {
        "health_risk": "Air quality is considered satisfactory. Air pollution poses little or no risk.",
        "most_affected": "None",
        "recommendations": [
            "Enjoy outdoor activities as normal.",
            "It's a great day to be active outside.",
            "No precautions are necessary for the general public.",
        ],
        "preventive_measures": [
            "Keep monitoring AQI if you are sensitive to pollution.",
            "Maintain good ventilation at home.",
        ],
    },
    "Moderate": {
        "health_risk": "Air quality is acceptable. However, there may be a risk for a small number of people who are unusually sensitive to air pollution.",
        "most_affected": "Unusually sensitive individuals (asthma, allergies).",
        "recommendations": [
            "Sensitive individuals should consider limiting prolonged outdoor exertion.",
            "General public can continue normal outdoor activities.",
            "Watch for symptoms such as coughing or shortness of breath.",
        ],
        "preventive_measures": [
            "Keep windows open for ventilation unless you are sensitive to pollen/dust.",
            "Sensitive groups should carry prescribed medication (e.g. inhalers).",
        ],
    },
    "Unhealthy for Sensitive Groups": {
        "health_risk": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
        "most_affected": "Children, elderly, people with heart or lung disease, pregnant women.",
        "recommendations": [
            "Sensitive groups should reduce prolonged or heavy outdoor exertion.",
            "Consider wearing a high-quality mask (N95/FFP2) if sensitive and going outside.",
            "Keep quick-relief medication accessible if you have asthma.",
        ],
        "preventive_measures": [
            "Keep windows closed during peak pollution hours.",
            "Use an indoor air purifier if available.",
            "Monitor local air quality updates regularly.",
        ],
    },
    "Unhealthy": {
        "health_risk": "Everyone may begin to experience health effects. Sensitive groups may experience more serious effects.",
        "most_affected": "Children, elderly, people with respiratory or cardiovascular conditions, outdoor workers.",
        "recommendations": [
            "Avoid prolonged outdoor activities.",
            "Wear a high-quality mask (N95/FFP2) when going outside.",
            "Sensitive groups should remain indoors as much as possible.",
            "Reschedule strenuous outdoor exercise.",
        ],
        "preventive_measures": [
            "Keep windows and doors closed.",
            "Use an indoor air purifier if available.",
            "Avoid busy roads and high-traffic areas.",
            "Stay hydrated and monitor for respiratory symptoms.",
        ],
    },
    "Very Unhealthy": {
        "health_risk": "Health alert: everyone may experience more serious health effects. This triggers a health alert.",
        "most_affected": "Entire population, especially children, elderly, and those with existing conditions.",
        "recommendations": [
            "Avoid all outdoor physical activity.",
            "Sensitive groups should remain indoors at all times.",
            "Wear an N95/FFP2 mask if you must go outside.",
            "Seek medical attention if respiratory symptoms develop.",
        ],
        "preventive_measures": [
            "Keep all windows and doors sealed.",
            "Run air purifiers continuously if available.",
            "Avoid any outdoor exercise or exertion.",
            "Check on vulnerable family members and neighbours.",
        ],
    },
    "Hazardous": {
        "health_risk": "Health emergency: the entire population is likely to be affected. Serious risk of respiratory and cardiovascular effects.",
        "most_affected": "Everyone, with the highest risk to children, elderly and people with pre-existing conditions.",
        "recommendations": [
            "Remain indoors and keep activity levels as low as possible.",
            "Avoid outdoor exposure entirely.",
            "Wear a high-quality mask (N95/FFP2) even for brief outdoor exposure.",
            "Seek immediate medical attention if severe respiratory symptoms develop.",
        ],
        "preventive_measures": [
            "Seal windows and doors; use air purifiers continuously.",
            "Follow local emergency air quality advisories.",
            "Avoid any physical exertion, indoors or outdoors.",
            "Evacuate the area if advised by local authorities.",
        ],
    },
}


def get_recommendations(category: str) -> dict:
    """
    Return the full recommendation package for a given AQI category.
    Falls back to 'Moderate' guidance if an unknown category is passed in,
    so the UI never breaks on unexpected input.
    """
    return RECOMMENDATIONS.get(category, RECOMMENDATIONS["Moderate"])
