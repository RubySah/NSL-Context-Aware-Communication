# src/context/scene_prompts.py
"""Prompts for two-stage CLIP scene understanding."""

# Stage 1: three descriptions per environment, averaged into one embedding.
SCENE_PROMPTS = {
    "Hospital / Clinic": [
        "A hospital room with a patient bed and medical equipment",
        "A clinic where doctors and nurses treat patients",
        "A pharmacy or hospital waiting area with medicine and chairs",
    ],
    "Classroom / Library": [
        "A school classroom with desks and a blackboard",
        "A library with bookshelves and students reading",
        "A study room with tables, books and a whiteboard",
    ],
    "Home": [
        "A home living room with a sofa and television",
        "A home bedroom with a bed and wardrobe",
        "A home kitchen with a dining table, stove and utensils",
    ],
    "Street / Outdoor": [
        "An urban street with cars and pedestrians",
        "A roadside area with vehicles and traffic signs",
        "An outdoor public space with buildings and sidewalks",
    ],
    "Office / Public Place": [
        "An office workspace with computers and desks",
        "A reception area with a desk and waiting chairs",
        "A shop or restaurant with a counter and tables",
    ],
}

# Stage 2: six objects per environment, checked only for the scene stage 1 picked.
SCENE_OBJECTS = {
    "Hospital / Clinic": [
        "a hospital bed", "a wheelchair", "an IV drip stand",
        "a patient monitor", "a privacy curtain", "a medicine cabinet",
    ],
    "Classroom / Library": [
        "a blackboard", "a whiteboard", "student desks",
        "bookshelves", "a projector screen", "classroom chairs",
    ],
    "Home": [
        "a sofa", "a television", "a bed",
        "a dining table", "curtains", "a refrigerator",
    ],
    "Street / Outdoor": [
        "a road", "cars", "motorcycles",
        "buses", "traffic lights", "buildings",
    ],
    "Office / Public Place": [
        "a desktop computer", "an office desk", "an office chair",
        "a reception desk", "waiting chairs", "a filing cabinet",
    ],
}

# Wrapper used for every object name before CLIP encodes it.
OBJECT_TEMPLATE = "a photo of {}."