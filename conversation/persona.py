"""Wale, the voice of Adugbo AI: greetings, thanks, and replying in Nigerian Pidgin.

Everything here is plain rules, not AI calls, so a "hi" costs nothing and can't be abused
to spend credit. The AI is only told which language to answer in (see advice_engine).
"""
import re

NAME = "Wale"

# --- language --------------------------------------------------------------------------------
# Words and phrases that are clearly Nigerian Pidgin and rare in standard English.
_PIDGIN = re.compile(
    r"\b(wetin|dey|abeg|una|wey|sabi|wahala|comot|abi|nawa|oya|shey|sebi|sef|jare|jor|ehen|pikin|omo|"
    r"how far|how body|how much be|i wan|e wan|you wan|e be|na im|na so|no be|make i|make we|make una|"
    r"i go like|how much e be|e get|e no|e don|una dey|wan rent|dem dey|na)\b",
    re.IGNORECASE,
)

# Pidgin has no fixed spelling: fold the common ways people type the same word into one form.
_SPELLINGS = [
    (re.compile(r"\bh(?:o)?w\s*fa(?:r|h|rr)?\b"), "how far"),  # howfar, how fa, hwfar, howfah
    (re.compile(r"\bw(?:h)?(?:e|a)t+in\b"), "wetin"),  # watin, wettin, whetin
    (re.compile(r"\bhow\s*bod(?:y|i)\b"), "how body"),  # howbody, how bodi
    (re.compile(r"\babeg+[iy]?\b"), "abeg"),  # abegg, abegi
    (re.compile(r"\bwahal+a+\b"), "wahala"),
    (re.compile(r"\bdey+\b"), "dey"),
]


def normalise(text: str) -> str:
    out = (text or "").lower()
    for pattern, canonical in _SPELLINGS:
        out = pattern.sub(canonical, out)
    return out


def is_pidgin(text: str) -> bool:
    return bool(_PIDGIN.search(normalise(text)))


def language_of(text: str) -> str:
    """'pcm' (Nigerian Pidgin) or 'en'."""
    return "pcm" if is_pidgin(text) else "en"


# --- greetings and thanks --------------------------------------------------------------------
_CORE_GREETING = {
    "hi", "hii", "hiii", "hello", "helo", "hey", "heya", "hiya", "yo", "sup", "wassup", "wasup",
    "howdy", "greetings", "morning", "afternoon", "evening", "bawo", "kaaro", "kaasan", "kaale",
    "ekaaro", "ekaasan", "ekaale", "kedu", "sannu", "ndewo", "salam", "salaam", "far", "dey",
}
# Words that may sit around a greeting without turning it into a question.
_FILLER = {
    "good", "how", "are", "you", "u", "ya", "doing", "is", "it", "going", "what", "whats", "up", "body",
    "wetin", "happen", "una", "ni", "e", "o", "oh", "there", "wale", "adugbo", "ai", "bot", "sir", "ma",
    "boss", "my", "guy", "friend", "everyone", "all", "man", "today", "alaikum", "alaykum", "nice", "to",
    "meet", "im", "i", "am", "and", "day", "the", "dear", "bro", "sis", "fam", "oga",
}
_THANKS = {"thanks", "thank", "thx", "tnx", "ty", "appreciate", "appreciated", "nagode", "ese", "daalu", "you", "u",
           "so", "much", "a", "lot", "wale", "o", "oh", "jare", "well", "done", "great", "nice", "cool", "ok", "okay",
           "alright", "abeg", "e", "don", "dey", "sir", "boss", "bro", "man"}


def _words(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z\s]", " ", normalise(text).replace("'", "").replace("ẹ", "e").replace("à", "a")
                     .replace("á", "a").replace("ọ", "o").replace("ò", "o").replace("ó", "o"))
    return cleaned.split()


def is_greeting(text: str) -> bool:
    """A message that is only a greeting ("hi", "good morning Wale", "how far"), not a question with a greeting."""
    words = _words(text)
    if not words or len(words) > 8:
        return False
    if not all(w in _CORE_GREETING or w in _FILLER for w in words):
        return False
    joined = " ".join(words)
    return (any(w in _CORE_GREETING for w in words)
            or any(p in joined for p in ("how are you", "how you", "how body", "how una")))


def is_thanks(text: str) -> bool:
    words = _words(text)
    if not words or len(words) > 6:
        return False
    starts = {"thanks", "thank", "thx", "tnx", "ty", "nagode", "ese", "daalu", "appreciate", "appreciated"}
    return any(w in starts for w in words) and all(w in _THANKS for w in words)


# --- what Wale says --------------------------------------------------------------------------
_EXAMPLES = {
    "en": ["2 bedroom in Yaba", "What can ₦3M a year get me in Ikeja?", "Compare Surulere and Gbagada"],
    "pcm": ["How much be 2 bedroom for Yaba?", "Wetin ₦3M fit get me for Ikeja?", "Compare Surulere and Gbagada"],
}

_GREETING = {
    "en": {
        "title": f"Hello! I'm {NAME}, your Adugbo AI.",
        "body": "I help you see what rent really costs in Lagos, using real listings gathered every week.",
        "can": [
            "Show rent prices for any area and house size",
            "Check what your budget can get you, and where",
            "Compare areas side by side",
            "Give practical renting tips: agent fees, deposits, what to check before you pay",
        ],
        "close": "What are you looking for?",
    },
    "pcm": {
        "title": f"How far! I be {NAME}, your Adugbo AI.",
        "body": "I dey help you know how much rent really be for Lagos, from real listings wey we dey gather every week.",
        "can": [
            "Show you rent price for any area and house size",
            "Check wetin your budget fit get you, and for where",
            "Compare areas side by side",
            "Give you correct renting advice: agent fee, caution fee, wetin to check before you pay",
        ],
        "close": "Wetin you dey find?",
    },
}

_THANKS_REPLY = {
    "en": f"You're welcome! {NAME} is here whenever you need to check rent in Lagos.",
    "pcm": f"No wahala at all! {NAME} dey here anytime you wan check rent for Lagos.",
}

OUT_OF_SCOPE = {
    "en": (f"I'm {NAME}, your Adugbo AI, and I can only help with renting in Lagos: prices by area and size, "
           "budgets, comparing areas, and practical renting advice. Try something like \"2 bedroom rent in Yaba\"."),
    "pcm": (f"I be {NAME}, your Adugbo AI. Na only rent for Lagos I fit help you with: price for area and house size, "
            "wetin your budget fit get, compare areas, and renting advice. Try ask me \"How much be 2 bedroom for Yaba?\""),
}

# One friendly line above listing-based answers when the user wrote in Pidgin. The figures stay as they are.
PIDGIN_LEADS = {
    "success": "See wetin rent dey cost, from real listings wey we gather:",
    "low_data": "Listing no plenty for this one, so take am as rough guide:",
    "no_data": "I no see listing wey match am for now. Check these options:",
    "comparison": "Make we compare dem side by side:",
    "budget": "See wetin your money fit get you:",
}

CLARIFY_AREA = {"en": "Which area are you looking for?", "pcm": "Which area you dey find house for?"}


def greeting(language: str) -> dict:
    g = _GREETING.get(language, _GREETING["en"])
    examples = _EXAMPLES.get(language, _EXAMPLES["en"])
    text = "\n".join([g["title"], g["body"], *[f"- {c}" for c in g["can"]], g["close"]])
    return {**g, "examples": examples, "text": text}


def thanks(language: str) -> dict:
    msg = _THANKS_REPLY.get(language, _THANKS_REPLY["en"])
    return {"title": msg, "body": "", "can": [], "close": "", "examples": _EXAMPLES.get(language, _EXAMPLES["en"]), "text": msg}


def pidgin_lead(answer: dict) -> str | None:
    if answer.get("comparison"):
        return PIDGIN_LEADS["comparison"]
    if answer.get("budget") and answer.get("status") in ("success", "low_data"):
        return PIDGIN_LEADS["budget"]
    return PIDGIN_LEADS.get(answer.get("status", ""))
