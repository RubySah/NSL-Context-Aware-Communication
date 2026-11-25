# src/fusion/task_domains.py
"""Static task-domain tags for the 52 signs and the 5 scenes."""

SIGN_DOMAINS = {
    "Ambulance": "emergency", "Help": "emergency",
    "Bandage": "medical", "Cough": "medical", "Doctor": "medical",
    "Fever": "medical", "Headache": "medical", "Hospital": "medical",
    "Injection": "medical", "Medicine": "medical", "Nurse": "medical",
    "Sick": "medical",
    "Book": "education", "Read": "education",
    "What": "information", "When": "information", "Where": "information",
    "Who": "information", "Time": "information", "Here": "information",
    "Left": "information", "Right": "information",
    # Everything else (pronouns, family, verbs, days, yes/no...) is "general".
}

# Keys match SceneDetector's lowercase environment names.
SCENE_DOMAINS = {
    "hospital / clinic": "medical",
    "classroom / library": "education",
    "home": "general",
    "street / outdoor": "emergency",
    "office / public place": "information",
}

OBJECT_DOMAINS = {
    "hospital bed": "medical", "wheelchair": "medical", "iv drip stand": "medical",
    "patient monitor": "medical", "medicine cabinet": "medical",
    "blackboard": "education", "whiteboard": "education", "student desks": "education",
    "reception desk": "information",
}