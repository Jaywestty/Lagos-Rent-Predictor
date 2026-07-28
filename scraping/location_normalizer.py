import json
import re
from pathlib import Path

_KNOWN_AREAS_PATH = Path(__file__).resolve().parent.parent / "data" / "known_areas.json"


def _load_known_areas():
    if not _KNOWN_AREAS_PATH.exists():
        return []
    with open(_KNOWN_AREAS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [entry["area"] for entry in data]


KNOWN_AREAS = _load_known_areas()
_AREAS_BY_TOKEN_LENGTH = sorted(KNOWN_AREAS, key=lambda a: -len(a.split()))

_ADMIN_SUFFIXES_PATH = Path(__file__).resolve().parent.parent / "data" / "known_admin_suffixes.json"

NOISE_TOKENS = {
    "off", "by", "no", "no.", "str", "st", "st.", "road", "rd", "rd.",
    "street", "estate", "behind", "junction", "close", "cl", "avenue",
    "ave", "opposite", "opp", "beside", "along", "via", "gra", "phase",
    "plot", "block", "house", "flat", "suite", "floor",
}

NOISE_PHRASES = [
    "bus stop", "by road", "opposite to", "beside the", "along the",
    "close to", "behind the", "off road", "by junction",
]

_UNIT_NUMBER_PATTERN = re.compile(r"\b\d+[a-z]?\b", flags=re.IGNORECASE)
_PUNCTUATION_PATTERN = re.compile(r"[.,'\u2019]")
_WHITESPACE_PATTERN = re.compile(r"\s+")

def _load_admin_suffixes():
    if not _ADMIN_SUFFIXES_PATH.exists():
        return set()
    with open(_ADMIN_SUFFIXES_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f))


ADMIN_SUFFIXES = _load_admin_suffixes()


def _strip_known_admin_suffix(text):
    lowered = text.lower()
    for suffix in sorted(ADMIN_SUFFIXES, key=lambda s: -len(s.split())):
        pattern = r"(?:^|\s)" + re.escape(suffix) + r"$"
        match = re.search(pattern, lowered)
        if match:
            return text[: match.start()].strip()
    return text


def _strip_noise_tokens(text):
    lowered = text.lower()
    for phrase in NOISE_PHRASES:
        lowered = lowered.replace(phrase, " ")

    cleaned = _PUNCTUATION_PATTERN.sub(" ", lowered)
    cleaned = _UNIT_NUMBER_PATTERN.sub(" ", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    tokens = [t for t in cleaned.split() if t not in NOISE_TOKENS]
    return " ".join(tokens).title()

def _collapse_repeated_phrase(text):
    tokens = text.split()
    n = len(tokens)
    if n >= 2 and n % 2 == 0:
        half = n // 2
        if tokens[:half] == tokens[half:]:
            return " ".join(tokens[:half])
    return text


def _strip_trailing_lagos(text):
    return re.sub(r",?\s*lagos\s*$", "", text.strip(), flags=re.IGNORECASE).strip()


def _match_known_area_suffix(text):
    cleaned = _strip_noise_tokens(text)
    lowered = cleaned.lower()
    for area in _AREAS_BY_TOKEN_LENGTH:
        pattern = r"(?:^|\s)" + re.escape(area.lower()) + r"$"
        match = re.search(pattern, lowered)
        if match:
            remainder = cleaned[: match.start()].strip()
            return area, remainder

    cleaned = _strip_known_admin_suffix(text)
    cleaned = _strip_noise_tokens(cleaned)
    lowered = cleaned.lower()
    for area in _AREAS_BY_TOKEN_LENGTH:
        pattern = r"(?:^|\s)" + re.escape(area.lower()) + r"$"
        match = re.search(pattern, lowered)
        if match:
            remainder = cleaned[: match.start()].strip()
            return area, remainder

    return None, text

def normalize_location(raw_location):
    if not raw_location or not isinstance(raw_location, str):
        return {
            "location_raw": raw_location,
            "area": None,
            "subarea": None,
            "needs_review": True,
        }

    cleaned = _strip_trailing_lagos(raw_location)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    parts = [_collapse_repeated_phrase(p) for p in parts]

    if len(parts) >= 2:
        area_candidate = parts[-1]
        subarea_candidate = ", ".join(parts[:-1])
        matched_area, _ = _match_known_area_suffix(area_candidate)
        area = matched_area if matched_area else area_candidate
        needs_review = matched_area is None
    elif len(parts) == 1:
        matched_area, remainder = _match_known_area_suffix(parts[0])
        if matched_area:
            area = matched_area
            subarea_candidate = remainder if remainder else None
            needs_review = False
        else:
            area = parts[0]
            subarea_candidate = None
            needs_review = True
    else:
        return {
            "location_raw": raw_location,
            "area": None,
            "subarea": None,
            "needs_review": True,
        }

    subarea = subarea_candidate if subarea_candidate else None

    return {
        "location_raw": raw_location,
        "area": area.title() if area else None,
        "subarea": subarea.title() if subarea else None,
        "needs_review": needs_review,
    }