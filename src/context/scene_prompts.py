"""Prompts for two-stage CLIP scene understanding."""

# Stage 1: three descriptions per environment, averaged into one embedding.
SCENE_PROMPTS = {
    "Hospital": [
        "A hospital room with a patient bed and medical equipment",
        "A medical facility where doctors and nurses provide care",
        "A hospital waiting room with chairs and a reception desk",
    ],
    "Classroom": [
        "A school classroom with desks and a blackboard",
        "A classroom with student chairs, books and school bags",
        "An educational room with a teacher's desk and whiteboard",
    ],
    "Home": [
        "A residential living room with a sofa and television",
        "A home bedroom with a bed and wardrobe",
        "A household interior with furniture and family belongings",
    ],
    "Outdoor / Street": [
        "An urban street with cars and pedestrians",
        "A roadside area with vehicles and traffic signs",
        "An outdoor public space with buildings and sidewalks",
    ],
    "Office": [
        "An office workspace with computers and desks",
        "A professional meeting room with a conference table",
        "An office where employees work at computer stations",
    ],
    "Kitchen / Dining Area": [
        "A household kitchen with a refrigerator and cooking stove",
        "A kitchen with cooking utensils, pots and food ingredients",
        "A dining area with a table, plates and chairs",
    ],
    "Shop / Market": [
        "A retail shop with product shelves and display counters",
        "A market with shopping baskets and product packages",
        "A shop with a cash register and price tags",
    ],
    "Library / Study Room": [
        "A library with bookshelves and rows of books",
        "A study room with tables, chairs and reading lamps",
        "A library with computers and study materials",
    ],
    "Pharmacy / Medical Shop": [
        "A pharmacy with shelves full of medicine boxes",
        "A medical shop with medicine bottles and prescriptions",
        "A pharmacy counter where medicines are sold",
    ],
    "Waiting Room / Reception Area": [
        "A reception area with a desk and waiting chairs",
        "A waiting room with visitor sofas and a registration counter",
        "A reception area with a computer and information board",
    ],
    "Restaurant / Cafe": [
        "A restaurant dining area with tables and chairs",
        "A cafe with food, menus and dining tables",
        "A restaurant with a serving counter and food trays",
    ],
    "Community Center / Gathering Place": [
        "A community hall with rows of chairs and a stage",
        "A gathering place with a microphone and speakers",
        "An event hall with a podium and projector screen",
    ],
}

# Stage 2: six objects per environment, checked only for the scene stage 1 picked.
SCENE_OBJECTS = {
    "Hospital": [
        "a hospital bed", "an IV drip stand", "a patient monitor",
        "a stethoscope", "a wheelchair", "medical equipment",
    ],
    "Classroom": [
        "a blackboard", "a whiteboard", "student desks",
        "classroom chairs", "books", "a projector",
    ],
    "Home": [
        "a sofa", "a television", "a coffee table",
        "a dining table", "curtains", "a bookshelf",
    ],
    "Outdoor / Street": [
        "a road", "cars", "motorcycles",
        "buses", "traffic lights", "pedestrians",
    ],
    "Office": [
        "a desktop computer", "a laptop", "an office desk",
        "an office chair", "a printer", "a meeting table",
    ],
    "Kitchen / Dining Area": [
        "a refrigerator", "a cooking stove", "cooking pots",
        "a frying pan", "a dining table", "plates and bowls",
    ],
    "Shop / Market": [
        "product shelves", "shopping baskets", "a cash register",
        "a display counter", "price tags", "product packages",
    ],
    "Library / Study Room": [
        "bookshelves", "books", "study tables",
        "reading chairs", "study lamps", "a computer",
    ],
    "Pharmacy / Medical Shop": [
        "medicine shelves", "medicine boxes", "medicine bottles",
        "prescription papers", "a pharmacist counter", "syringes",
    ],
    "Waiting Room / Reception Area": [
        "a reception desk", "waiting chairs", "a visitor sofa",
        "a registration counter", "an information board", "a token display screen",
    ],
    "Restaurant / Cafe": [
        "dining tables", "dining chairs", "food plates",
        "drinking glasses", "menu cards", "a restaurant counter",
    ],
    "Community Center / Gathering Place": [
        "rows of chairs", "a stage", "a microphone",
        "speakers", "a podium", "a projector screen",
    ],
}

# Wrapper used for every object name before CLIP encodes it.
OBJECT_TEMPLATE = "a photo of {}."