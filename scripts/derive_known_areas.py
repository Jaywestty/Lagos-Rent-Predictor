import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

MIN_AREA_FREQUENCY = 2


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


def match_single_part_against_known(single_part_values, known_areas):
    areas_by_word_count = sorted(known_areas, key=lambda a: -len(a.split()))
    matched_counter = Counter()
    unmatched = []

    for value in single_part_values:
        lowered = value.lower()
        matched = False
        for area in areas_by_word_count:
            pattern = r"(?:^|\s)" + re.escape(area.lower()) + r"$"
            if re.search(pattern, lowered):
                matched_counter[area] += 1
                matched = True
                break
        if not matched:
            unmatched.append(value)

    return matched_counter, unmatched


def find_contains_fallback_match(unmatched_values, known_areas):
    areas_by_word_count = sorted(known_areas, key=lambda a: -len(a.split()))
    fallback_matches = {}
    still_unmatched = []

    for value in unmatched_values:
        lowered = value.lower()
        found_area = None
        for area in areas_by_word_count:
            pattern = r"(?:^|\s)" + re.escape(area.lower()) + r"(?:\s|$)"
            if re.search(pattern, lowered):
                found_area = area
                break
        if found_area:
            fallback_matches[value] = found_area
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
    parser.add_argument("--area-aliases", default="data/area_aliases.json")
    parser.add_argument("--min-frequency", type=int, default=MIN_AREA_FREQUENCY)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    multi_part_counter, single_part_values = extract_area_candidates(df[args.location_column])

    manual_overrides = load_manual_overrides(Path(args.manual_overrides))
    area_aliases = load_area_aliases(Path(args.area_aliases))

    threshold_passed = {area for area, count in multi_part_counter.items() if count >= args.min_frequency}
    known_areas = threshold_passed | manual_overrides

    matched_counter, unmatched = match_single_part_against_known(single_part_values, known_areas)
    fallback_matches, still_unmatched = find_contains_fallback_match(unmatched, known_areas)

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