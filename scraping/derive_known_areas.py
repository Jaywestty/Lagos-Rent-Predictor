import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

MIN_AREA_FREQUENCY = 2

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


def load_admin_suffixes(path):
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def strip_known_admin_suffix(text, admin_suffixes):
    lowered = text.lower()
    for suffix in sorted(admin_suffixes, key=lambda s: -len(s.split())):
        pattern = r"(?:^|\s)" + re.escape(suffix) + r"$"
        match = re.search(pattern, lowered)
        if match:
            return text[: match.start()].strip()
    return text


def strip_noise_tokens(text):
    lowered = text.lower()
    for phrase in NOISE_PHRASES:
        lowered = lowered.replace(phrase, " ")

    cleaned = _PUNCTUATION_PATTERN.sub(" ", lowered)
    cleaned = _UNIT_NUMBER_PATTERN.sub(" ", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    tokens = [t for t in cleaned.split() if t not in NOISE_TOKENS]
    return " ".join(tokens).title()

def strip_trailing_lagos(text):
    return re.sub(r",?\s*lagos\s*$", "", text.strip(), flags=re.IGNORECASE).strip()


def collapse_adjacent_duplicate_word(text):
    tokens = text.split()
    collapsed = []
    for token in tokens:
        if collapsed and collapsed[-1].lower() == token.lower():
            continue
        collapsed.append(token)
    return " ".join(collapsed)


def load_manual_overrides(path):
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data)


def load_area_aliases(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_area_aliases(counter, aliases):
    folded = Counter()
    for area, count in counter.items():
        canonical = aliases.get(area, area)
        folded[canonical] += count
    return folded


def extract_area_candidates(location_series):
    multi_part_counter = Counter()
    single_part_values = []

    for raw in location_series.dropna():
        cleaned = strip_trailing_lagos(raw)
        parts = [collapse_adjacent_duplicate_word(p.strip()) for p in cleaned.split(",") if p.strip()]

        if len(parts) >= 2:
            multi_part_counter[parts[-1]] += 1
        elif len(parts) == 1:
            single_part_values.append(parts[0])

    return multi_part_counter, single_part_values


def _try_end_match(lowered_text, areas_by_word_count):
    for area in areas_by_word_count:
        pattern = r"(?:^|\s)" + re.escape(area.lower()) + r"$"
        if re.search(pattern, lowered_text):
            return area
    return None


def _try_contains_match(lowered_text, areas_by_word_count):
    best_area = None
    best_position = -1
    for area in areas_by_word_count:
        pattern = r"(?:^|\s)" + re.escape(area.lower()) + r"(?:\s|$)"
        for match in re.finditer(pattern, lowered_text):
            if match.start() > best_position:
                best_position = match.start()
                best_area = area
    return best_area


def match_single_part_against_known(single_part_values, known_areas, admin_suffixes):
    areas_by_word_count = sorted(known_areas, key=lambda a: -len(a.split()))
    matched_counter = Counter()
    unmatched = []

    for value in single_part_values:
        noise_only = strip_noise_tokens(value)
        area = _try_end_match(noise_only.lower(), areas_by_word_count)

        if not area:
            admin_stripped = strip_known_admin_suffix(value, admin_suffixes)
            admin_stripped = strip_noise_tokens(admin_stripped)
            area = _try_end_match(admin_stripped.lower(), areas_by_word_count)

        if area:
            matched_counter[area] += 1
        else:
            unmatched.append(value)

    return matched_counter, unmatched


def find_contains_fallback_match(unmatched_values, known_areas, admin_suffixes):
    areas_by_word_count = sorted(known_areas, key=lambda a: -len(a.split()))
    fallback_matches = {}
    still_unmatched = []

    for value in unmatched_values:
        noise_only = strip_noise_tokens(value)
        area = _try_contains_match(noise_only.lower(), areas_by_word_count)

        if not area:
            admin_stripped = strip_known_admin_suffix(value, admin_suffixes)
            admin_stripped = strip_noise_tokens(admin_stripped)
            area = _try_contains_match(admin_stripped.lower(), areas_by_word_count)

        if area:
            fallback_matches[value] = area
        else:
            still_unmatched.append(value)

    return fallback_matches, still_unmatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/lagos_properties.csv")
    parser.add_argument("--location-column", default="location")
    parser.add_argument("--known-areas-out", default="data/known_areas.json")
    parser.add_argument("--review-out", default="data/location_review_needed.csv")
    parser.add_argument("--manual-overrides", default="data/manual_known_areas.json")
    parser.add_argument("--admin-suffixes", default="data/known_admin_suffixes.json")
    parser.add_argument("--area-aliases", default="data/area_aliases.json")
    parser.add_argument("--min-frequency", type=int, default=MIN_AREA_FREQUENCY)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    multi_part_counter, single_part_values = extract_area_candidates(df[args.location_column])

    manual_overrides = load_manual_overrides(Path(args.manual_overrides))
    area_aliases = load_area_aliases(Path(args.area_aliases))

    admin_suffixes = load_admin_suffixes(Path(args.admin_suffixes))

    threshold_passed = {area for area, count in multi_part_counter.items() if count >= args.min_frequency}
    known_areas = threshold_passed | manual_overrides

    matched_counter, unmatched = match_single_part_against_known(single_part_values, known_areas, admin_suffixes)
    fallback_matches, still_unmatched = find_contains_fallback_match(unmatched, known_areas, admin_suffixes)

    combined_counter = Counter()
    for area in known_areas:
        if area in multi_part_counter:
            combined_counter[area] += multi_part_counter[area]
    combined_counter += matched_counter
    for area in fallback_matches.values():
        combined_counter[area] += 1

    combined_counter = apply_area_aliases(combined_counter, area_aliases)

    known_areas_final = sorted(
        [{"area": area, "count": count} for area, count in combined_counter.items()],
        key=lambda r: -r["count"],
    )

    below_threshold = [
        {"area": area, "count": count}
        for area, count in multi_part_counter.items()
        if count < args.min_frequency and area not in manual_overrides
    ]

    Path(args.known_areas_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.known_areas_out, "w", encoding="utf-8") as f:
        json.dump(known_areas_final, f, indent=2)

    review_map = {}
    for value in sorted(set(still_unmatched)):
        review_map[value] = ""
    for value, area in sorted(fallback_matches.items()):
        review_map[value] = area
    for entry in below_threshold:
        review_map.setdefault(entry["area"], "")

    review_rows = [
        {"raw_value_needing_review": value, "fallback_area_guess": guess}
        for value, guess in sorted(review_map.items())
    ]

    pd.DataFrame(review_rows).to_csv(args.review_out, index=False)

    print(f"Derived {len(known_areas_final)} known areas from data.")
    print(f"Known areas written to {args.known_areas_out}")
    print(f"{len(review_rows)} values need manual review, written to {args.review_out}")
    print(f"{len(fallback_matches)} matched via contains-fallback (flagged for confirmation, not silently trusted)")


if __name__ == "__main__":
    main()