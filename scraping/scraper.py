"""
lagos_property_scraper.py

Scraper for Lagos house listings (rent + sale) from:
  - NigeriaPropertyCentre (nigeriapropertycentre.com)
  - PropertyPro.ng (propertypro.ng)

Design notes:
  - Search/listing pages are used only to discover candidate listing URLs
    via stable URL patterns. They are not parsed for structured fields.
  - Every structured field (price, bedrooms, bathrooms, toilets, parking,
    title) is extracted from the listing's own detail page. This is
    necessary because NPC's search-results cards render each listing
    twice, once with labelled counts ("4 Beds 4 Baths 5 Toilets") and once
    as bare digits with no labels ("4 4 5"), so label-anchored extraction
    on the card silently drops fields whenever the bare-digit render is
    the one captured. PropertyPro's search-results card only shows beds
    and baths at all, so toilets can never be captured there regardless
    of parsing quality.
  - Parking is only ever present on NPC. PropertyPro does not expose a
    parking count on either its search cards or its detail pages, so this
    field is expected to remain null for every PropertyPro row.
  - Property type, area, subarea, and a bedrooms fallback are still parsed
    from the URL slug, since both sites encode these reliably there and
    it is independent of on-page text changes.
  - Each site is crawled across multiple property categories (houses,
    flats/apartments, self-contain, mini-flat) to avoid skewing the
    dataset toward duplex/detached-house listings only. NPC's
    flats-apartments category already includes studio and self-contained
    units within the same search results, so it does not need separate
    sub-categories. PropertyPro splits these out as distinct top-level
    categories in its own navigation, so they are crawled separately.
  - price_period is frequently absent from PropertyPro's own listing
    page for the primary listing (confirmed by inspecting a live page:
    the price renders as a bare "N18,000,000" with no period marker,
    while unrelated listings in that same page's "Recommended
    Properties" sidebar do show "/year"). This is a source-data gap,
    not a parsing bug — do not "fix" this by loosening the period
    regex, it will not help.
  - Resumable: checkpoint file written after every page, keyed by
    site + category + listing_type so different categories never share
    resume state.
  - Deduplicated by listing URL, appended incrementally to CSV.
  - Rate-limited with jitter and retry/backoff on every request, including
    each individual detail-page fetch.

Usage:
    python lagos_property_scraper.py --site npc --category houses --type rent --max-pages 20
    python lagos_property_scraper.py --site propertypro --category flat-apartment --type rent --max-pages 20
    python lagos_property_scraper.py --site all --category all --type all --max-pages 5
    python lagos_property_scraper.py --site npc --category houses --type rent --inspect   # debug mode, 1 page

Output:
    data/raw/lagos_properties.csv  (append mode, deduped by url)
    data/raw/.checkpoint_<site>_<category>_<type>.json  (resume state)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

OUTPUT_DIR = Path("data/raw")
OUTPUT_CSV = OUTPUT_DIR / "lagos_properties.csv"
CHECKPOINT_DIR = OUTPUT_DIR
REQUEST_TIMEOUT = 15
MIN_DELAY = 1.5
MAX_DELAY = 3.5
MAX_RETRIES = 3
BACKOFF_BASE = 2.0

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

SITE_CONFIGS = {
    "npc": {
        "categories": {
            "houses": {
                "rent": "https://nigeriapropertycentre.com/for-rent/houses/lagos/showtype",
                "sale": "https://nigeriapropertycentre.com/for-sale/houses/lagos/showtype",
            },
            "flats-apartments": {
                "rent": "https://nigeriapropertycentre.com/for-rent/flats-apartments/lagos/showtype",
                "sale": "https://nigeriapropertycentre.com/for-sale/flats-apartments/lagos/showtype",
            },
        },
        "detail_url_pattern": re.compile(
            r"/(for-rent|for-sale)/(?:houses|flats-apartments)/.+/\d{5,}-"
        ),
        "page_param": "page",
        "page_start": 1,
    },
    "propertypro": {
        "categories": {
            "house": {
                "rent": "https://propertypro.ng/property-for-rent/house/in/lagos",
                "sale": "https://propertypro.ng/property-for-sale/house/in/lagos",
            },
            "flat-apartment": {
                "rent": "https://propertypro.ng/property-for-rent/flat-apartment/in/lagos",
                "sale": "https://propertypro.ng/property-for-sale/flat-apartment/in/lagos",
            },
            "self-contain": {
                "rent": "https://propertypro.ng/property-for-rent/self-contain/in/lagos",
            },
            "mini-flat": {
                "rent": "https://propertypro.ng/property-for-rent/mini-flat/in/lagos",
            },
        },
        "detail_url_pattern": re.compile(r"/property/[a-z0-9-]+-[A-Za-z0-9]{5}$"),
        "page_param": "page",
        "page_start": 0,
        # Newest first, so a recency-limited run can stop once it reaches old listings.
        "recent_sort_query": "sort=postedOn&order=desc",
    },
}

# A recency-limited run stops a category after this many pages in a row
# contain no listing inside the window.
RECENT_STOP_AFTER_STALE_PAGES = 2

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Listing:
    source: str
    listing_type: str
    url: str
    title: str | None = None
    price_raw: str | None = None
    price_ngn: float | None = None
    price_period: str | None = None
    location: str | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    toilets: int | None = None
    parking: int | None = None
    property_type: str | None = None
    scraped_at: str = ""
    listed_at: str | None = None


CSV_FIELDS = [f.name for f in fields(Listing)]

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("lagos_scraper")

# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

class Fetcher:
    def __init__(self):
        self.session = requests.Session()

    def get(self, url: str) -> BeautifulSoup | None:
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept-Language": "en-NG,en;q=0.9",
        }
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    return BeautifulSoup(resp.text, "html.parser")
                if resp.status_code in (429, 503):
                    wait = BACKOFF_BASE ** attempt + random.uniform(1, 3)
                    log.warning(f"{resp.status_code} on {url}, backing off {wait:.1f}s "
                                f"(attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                log.warning(f"Unexpected status {resp.status_code} for {url}")
                return None
            except requests.RequestException as e:
                wait = BACKOFF_BASE ** attempt
                log.warning(f"Request error on {url}: {e} — retrying in {wait:.1f}s "
                            f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
        log.error(f"Giving up on {url} after {MAX_RETRIES} attempts")
        return None

    @staticmethod
    def polite_sleep():
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


# --------------------------------------------------------------------------
# Regex helpers
# --------------------------------------------------------------------------

PRICE_RE = re.compile(r"₦\s*([\d,]{4,})")
PERIOD_RE = re.compile(r"/\s?(yr|year|mo|month)", re.IGNORECASE)


def _clean_price(text: str) -> tuple[str | None, float | None, str | None]:
    m = PRICE_RE.search(text)
    if not m:
        return None, None, None
    raw = m.group(0).strip()
    numeric = float(m.group(1).replace(",", ""))
    period_m = PERIOD_RE.search(text)
    period = None
    if period_m:
        p = period_m.group(1).lower()
        period = "year" if p in ("yr", "year") else "month"
    return raw, numeric, period


def _infer_rent_period(text: str) -> str | None:
    if re.search(r"per\s*annum|/\s*annum|\bp\.?a\.?\b", text, re.IGNORECASE):
        return "year"
    if re.search(r"per\s*month|monthly|/\s*mo\b", text, re.IGNORECASE):
        return "month"
    return None


def _extract_labeled_count(text: str, label: str, filler: str = "") -> int | None:
    """
    Matches a count that can appear either as "N Label" or "Label N", since
    the same detail page often shows the same field both ways in different
    sections (a feature summary vs a details table). filler allows an
    optional word between the number and label, e.g. "2 spaces Parking".
    """
    filler_part = rf"(?:{filler}\s*)?" if filler else ""
    m = re.search(rf"(\d+)\s*{filler_part}{label}s?\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(rf"{label}s?\s*{filler_part}[:\-]?\s*(\d+)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _truncate_before_marker(text: str, marker: str) -> str:
    """
    Cuts text at the first occurrence of marker, used to exclude
    "similar/recommended properties" sections further down the page so
    their prices and counts are never mistaken for the target listing's.
    """
    idx = text.lower().find(marker.lower())
    return text[:idx] if idx != -1 else text

NPC_DATE_POSTED_RE = re.compile(r'"datePosted"\s*:\s*"([^"]+)"')
NPC_POSTED_TEXT_RE = re.compile(r"Posted\s+(\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4})")
PP_LAST_UPDATED_RE = re.compile(r"Last updated\s+(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?")
LAGOS_TZ = timezone(timedelta(hours=1))


def _parse_npc_listed_at(soup: BeautifulSoup) -> datetime | None:
    """NPC detail pages carry JSON-LD datePosted, with a visible "Posted 17 Nov 2026" fallback."""
    m = NPC_DATE_POSTED_RE.search(str(soup))
    if m:
        try:
            dt = datetime.fromisoformat(m.group(1))
            return dt if dt.tzinfo else dt.replace(tzinfo=LAGOS_TZ)
        except ValueError:
            pass
    m = NPC_POSTED_TEXT_RE.search(soup.get_text(" ", strip=True))
    if m:
        try:
            return datetime.strptime(m.group(1), "%d %b %Y").replace(tzinfo=LAGOS_TZ)
        except ValueError:
            pass
    return None


def _parse_propertypro_listed_at(soup: BeautifulSoup) -> datetime | None:
    """PropertyPro detail pages show "Last updated 2026-11-17 13:02:15.0"."""
    m = PP_LAST_UPDATED_RE.search(soup.get_text(" ", strip=True))
    if not m:
        return None
    try:
        stamp = f"{m.group(1)} {m.group(2) or '00:00:00'}"
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=LAGOS_TZ)
    except ValueError:
        return None


def _is_out_of_scope_listing(url: str) -> bool:
    """
    PropertyPro cross-lists shortlet units inside its regular rent search
    results. Shortlet pricing (daily/weekly) is not comparable to annual
    rent and must not be conflated with it downstream.
    """
    return "-for-shortlet-" in url


# --------------------------------------------------------------------------
# URL-based structured parsing
# --------------------------------------------------------------------------

def parse_npc_url(url: str) -> dict:
    parts = [p for p in urlparse(url).path.split("/") if p]
    result = {"property_type": None, "area": None, "subarea": None,
              "bedrooms": None, "title": None}
    if "lagos" in parts:
        li = parts.index("lagos")
        if li >= 1:
            result["property_type"] = parts[li - 1].replace("-", " ").title()
        if len(parts) > li + 1:
            result["area"] = parts[li + 1].replace("-", " ").title()
        if len(parts) > li + 2 and not re.match(r"^\d+-", parts[li + 2]):
            result["subarea"] = parts[li + 2].replace("-", " ").title()
    if parts:
        m = re.match(r"^\d+-(.+)$", parts[-1])
        if m:
            slug = m.group(1)
            result["title"] = slug.replace("-", " ").title()
            bed_m = re.search(r"(\d+)-bedroom", slug)
            if bed_m:
                result["bedrooms"] = int(bed_m.group(1))
    return result


def parse_propertypro_url(url: str) -> dict:
    parts = [p for p in urlparse(url).path.split("/") if p]
    result = {"property_type": None, "area": None, "bedrooms": None,
              "title": None, "pid": None}
    if not parts:
        return result
    slug = parts[-1]
    result["title"] = slug.rsplit("-", 1)[0].replace("-", " ").title()

    bed_m = re.search(r"^(\d+)-bedroom", slug)
    if bed_m:
        result["bedrooms"] = int(bed_m.group(1))

    type_m = re.search(r"bedroom-([a-z]+(?:-[a-z]+)*)-for-(?:rent|sale)-", slug)
    if type_m:
        result["property_type"] = type_m.group(1).title()
    elif re.search(r"^self-contain-for-(?:rent|sale)-", slug):
        result["property_type"] = "Self Contain"
    elif re.search(r"^mini-flat-for-(?:rent|sale)-", slug):
        result["property_type"] = "Mini Flat"

    loc_m = re.search(r"for-(?:rent|sale)-(.+)-lagos-[A-Za-z0-9]+$", slug, re.IGNORECASE)
    if loc_m:
        result["area"] = loc_m.group(1).replace("-", " ").title()

    pid_m = re.search(r"-([A-Za-z0-9]{5})$", slug)
    if pid_m:
        result["pid"] = pid_m.group(1)
    return result


URL_PARSERS = {
    "npc": parse_npc_url,
    "propertypro": parse_propertypro_url,
}


def _build_location_from_url(url_data: dict) -> str | None:
    parts = [url_data.get("subarea"), url_data.get("area")]
    parts = [p for p in parts if p]
    return f"{', '.join(parts)}, Lagos" if parts else None


# --------------------------------------------------------------------------
# Detail-page field extraction
# --------------------------------------------------------------------------

def parse_npc_detail_fields(soup: BeautifulSoup) -> dict:
    text = soup.get_text(" ", strip=True)
    text = _truncate_before_marker(text, "similar properties in")

    price_raw, price_ngn, period = _clean_price(text)
    if period is None:
        period = _infer_rent_period(text)

    h1 = soup.find("h1")

    return {
        "title": h1.get_text(strip=True) if h1 else None,
        "price_raw": price_raw,
        "price_ngn": price_ngn,
        "price_period": period,
        "bedrooms": _extract_labeled_count(text, "Bedroom"),
        "bathrooms": _extract_labeled_count(text, "Bathroom"),
        "toilets": _extract_labeled_count(text, "Toilet"),
        "parking": _extract_labeled_count(text, "Parking", filler="spaces?"),
        "listed_at": _parse_npc_listed_at(soup),
    }


def parse_propertypro_detail_fields(soup: BeautifulSoup) -> dict:
    text = soup.get_text(" ", strip=True)
    text = _truncate_before_marker(text, "recommended properties")

    price_raw, price_ngn, period = _clean_price(text)
    if period is None:
        period = _infer_rent_period(text)

    h1 = soup.find("h1")

    return {
        "title": h1.get_text(strip=True) if h1 else None,
        "price_raw": price_raw,
        "price_ngn": price_ngn,
        "price_period": period,
        "bedrooms": _extract_labeled_count(text, "Bed"),
        "bathrooms": _extract_labeled_count(text, "Bath"),
        "toilets": _extract_labeled_count(text, "Toilet"),
        "parking": _extract_labeled_count(text, "Parking", filler="spaces?"),
        "listed_at": _parse_propertypro_listed_at(soup),
    }


DETAIL_PARSERS = {
    "npc": parse_npc_detail_fields,
    "propertypro": parse_propertypro_detail_fields,
}


# --------------------------------------------------------------------------
# Search-page URL discovery
# --------------------------------------------------------------------------

def discover_listing_urls(soup: BeautifulSoup, base_url: str, url_pattern: re.Pattern) -> list[str]:
    hrefs = {a["href"] for a in soup.find_all("a", href=True) if url_pattern.search(a["href"])}
    return [urljoin(base_url, href) for href in hrefs]


def _detect_near_duplicates(listings: list[Listing]):
    """
    Same-page early warning only. This does not replace load_data.py's
    dedup pass, which compares across the full dataset and across runs.
    """
    seen = {}
    for item in listings:
        key = (item.bedrooms, item.location, item.price_ngn, item.property_type)
        if all(v is not None for v in key):
            if key in seen:
                log.warning(f"Possible duplicate: {item.url} matches {seen[key]} on beds/location/price.")
            else:
                seen[key] = item.url


# --------------------------------------------------------------------------
# Checkpointing + CSV persistence
# --------------------------------------------------------------------------

def checkpoint_path(site: str, category: str, listing_type: str) -> Path:
    return CHECKPOINT_DIR / f".checkpoint_{site}_{category}_{listing_type}.json"


def load_checkpoint(site: str, category: str, listing_type: str) -> dict:
    p = checkpoint_path(site, category, listing_type)
    if p.exists():
        return json.loads(p.read_text())
    return {"last_page": None, "seen_urls": []}


def save_checkpoint(site: str, category: str, listing_type: str, last_page: int, seen_urls: set[str]):
    p = checkpoint_path(site, category, listing_type)
    p.write_text(json.dumps({"last_page": last_page, "seen_urls": list(seen_urls)}))


def append_to_csv(listings: list[Listing], output_csv: Path = OUTPUT_CSV):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for listing in listings:
            writer.writerow(asdict(listing))


# --------------------------------------------------------------------------
# Page URL builder
# --------------------------------------------------------------------------

def build_page_url(base_url: str, page_param: str, page_num: int, page_start: int) -> str:
    if page_num == page_start:
        return base_url
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}{page_param}={page_num}"


# --------------------------------------------------------------------------
# Main crawl loop
# --------------------------------------------------------------------------

def _new_stats() -> dict:
    return {"discovered": 0, "kept": 0, "too_old": 0, "no_date": 0, "no_price": 0, "fetch_failed": 0}


def crawl(site: str, category: str, listing_type: str, max_pages: int, inspect: bool = False,
          since_days: int | None = None, output_csv: Path = OUTPUT_CSV,
          stats: dict | None = None) -> dict:
    """Crawl one site/category/type.

    With since_days set, the run is recency-limited: it starts at the newest
    page (ignoring the resume checkpoint), keeps only listings whose source
    date is within the window, and stops after RECENT_STOP_AFTER_STALE_PAGES
    pages in a row with nothing recent.
    """
    stats = stats if stats is not None else _new_stats()
    config = SITE_CONFIGS[site]
    category_urls = config["categories"].get(category)
    if category_urls is None:
        log.warning(f"[{site}] Unknown category '{category}', skipping.")
        return stats

    base_url = category_urls.get(listing_type)
    if base_url is None:
        log.info(f"[{site}/{category}/{listing_type}] Category does not support this "
                  f"listing type, skipping.")
        return stats

    recent_mode = since_days is not None
    cutoff = datetime.now(UTC) - timedelta(days=since_days) if recent_mode else None
    if recent_mode and config.get("recent_sort_query"):
        sep = "&" if "?" in base_url else "?"
        base_url = f"{base_url}{sep}{config['recent_sort_query']}"

    url_pattern = config["detail_url_pattern"]
    page_param = config["page_param"]
    page_start = config["page_start"]
    url_parser = URL_PARSERS[site]
    detail_parser = DETAIL_PARSERS[site]

    fetcher = Fetcher()
    if recent_mode:
        # Weekly runs always start from the newest page; the database upsert handles repeats.
        seen_urls: set[str] = set()
        start_page = page_start
    else:
        ckpt = load_checkpoint(site, category, listing_type)
        seen_urls = set(ckpt["seen_urls"])
        start_page = (ckpt["last_page"] + 1) if ckpt["last_page"] is not None else page_start

    if start_page > page_start:
        log.info(f"[{site}/{category}/{listing_type}] Resuming from page {start_page} "
                 f"({len(seen_urls)} listings already collected)")

    pages_to_run = 1 if inspect else max_pages
    total_new = 0
    inspected_count = 0
    stale_pages = 0

    for offset in range(pages_to_run):
        page_num = start_page + offset
        page_url = build_page_url(base_url, page_param, page_num, page_start)
        log.info(f"[{site}/{category}/{listing_type}] Fetching page {page_num}: {page_url}")

        soup = fetcher.get(page_url)
        if soup is None:
            log.warning(f"[{site}/{category}/{listing_type}] Skipping page {page_num} (fetch failed)")
            fetcher.polite_sleep()
            continue

        candidate_urls = discover_listing_urls(soup, page_url, url_pattern)
        if not candidate_urls:
            log.warning(f"[{site}/{category}/{listing_type}] No listing links found on page "
                        f"{page_num} — likely reached the end, or a structural change on the site.")
            break

        new_urls = [u for u in candidate_urls if u not in seen_urls]

        shortlet_urls = {u for u in new_urls if _is_out_of_scope_listing(u)}
        if shortlet_urls:
            log.info(f"[{site}/{category}/{listing_type}] Skipping {len(shortlet_urls)} "
                      f"shortlet listing(s) — out of scope (short-term rental, not annual rent).")
            seen_urls.update(shortlet_urls)
        new_urls = [u for u in new_urls if u not in shortlet_urls]

        page_listings = []
        recent_on_page = 0
        stats["discovered"] += len(new_urls)

        for listing_url in new_urls:
            fetcher.polite_sleep()
            detail_soup = fetcher.get(listing_url)
            if detail_soup is None:
                log.warning(f"Fetch failed for {listing_url}, will be retried on next run.")
                stats["fetch_failed"] += 1
                continue

            url_data = url_parser(listing_url)
            detail_fields = detail_parser(detail_soup)
            listed_at = detail_fields.get("listed_at")

            if recent_mode:
                seen_urls.add(listing_url)
                if listed_at is None:
                    stats["no_date"] += 1
                    log.info(f"No listing date for {listing_url}, skipping in recent mode.")
                    continue
                if listed_at < cutoff:
                    stats["too_old"] += 1
                    continue
                recent_on_page += 1

            if not detail_fields.get("price_ngn"):
                log.info(f"No price extracted for {listing_url}, dropping.")
                stats["no_price"] += 1
                continue

            listing = Listing(
                source=site,
                listing_type=listing_type,
                url=listing_url,
                title=detail_fields.get("title") or url_data.get("title"),
                price_raw=detail_fields.get("price_raw"),
                price_ngn=detail_fields.get("price_ngn"),
                price_period=detail_fields.get("price_period"),
                location=_build_location_from_url(url_data),
                bedrooms=detail_fields.get("bedrooms") or url_data.get("bedrooms"),
                bathrooms=detail_fields.get("bathrooms"),
                toilets=detail_fields.get("toilets"),
                parking=detail_fields.get("parking"),
                property_type=url_data.get("property_type"),
                scraped_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                listed_at=listed_at.isoformat() if listed_at else None,
            )

            if inspect:
                print(f"\n--- {listing_url} ---")
                print(json.dumps(asdict(listing), indent=2, ensure_ascii=False))
                inspected_count += 1
                if inspected_count >= 5:
                    print("\n(Inspect mode: stopping after 5 listings. Verify values look "
                          "correct before running a full crawl.)")
                    return
                continue

            page_listings.append(listing)
            seen_urls.add(listing_url)

        if inspect:
            continue

        if page_listings:
            _detect_near_duplicates(page_listings)
            append_to_csv(page_listings, output_csv)
            total_new += len(page_listings)
            stats["kept"] += len(page_listings)
            log.info(f"[{site}/{category}/{listing_type}] Page {page_num}: "
                     f"{len(page_listings)} new listings (total this run: {total_new})")
        else:
            log.info(f"[{site}/{category}/{listing_type}] Page {page_num}: no new listings "
                     f"(all {len(candidate_urls)} already seen — possible end of results)")

        if recent_mode:
            stale_pages = stale_pages + 1 if recent_on_page == 0 else 0
            if stale_pages >= RECENT_STOP_AFTER_STALE_PAGES:
                log.info(f"[{site}/{category}/{listing_type}] {stale_pages} pages in a row with nothing "
                         f"from the last {since_days} days, stopping.")
                break
        else:
            save_checkpoint(site, category, listing_type, page_num, seen_urls)
        fetcher.polite_sleep()

    log.info(f"[{site}/{category}/{listing_type}] Done. {total_new} new listings this run, "
             f"{len(seen_urls)} total collected so far.")
    return stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Scrape Lagos house listings (rent/sale).")
    parser.add_argument("--site", choices=["npc", "propertypro", "all"], default="all")
    parser.add_argument("--category", default="all",
                        help="Category to crawl (e.g. houses, flats-apartments, house, "
                             "flat-apartment, self-contain, mini-flat). 'all' crawls every "
                             "category defined for the selected site(s).")
    parser.add_argument("--type", choices=["rent", "sale", "all"], default="all")
    parser.add_argument("--max-pages", type=int, default=10,
                        help="Pages to crawl per site/category/type this run (resumable — run again to continue).")
    parser.add_argument("--inspect", action="store_true",
                        help="Fetch 1 page, print up to 5 sample extracted listings, exit. Use this first.")
    parser.add_argument("--since-days", type=int, default=None,
                        help="Only keep listings posted/updated in the last N days, newest pages first, "
                             "and stop when pages turn stale. Ignores the resume checkpoint.")
    parser.add_argument("--output", default=str(OUTPUT_CSV),
                        help="CSV to append to (default data/raw/lagos_properties.csv).")
    args = parser.parse_args(argv)
    stats = _new_stats()
    output_csv = Path(args.output)

    sites = list(SITE_CONFIGS.keys()) if args.site == "all" else [args.site]
    types = ["rent", "sale"] if args.type == "all" else [args.type]

    for site in sites:
        site_categories = list(SITE_CONFIGS[site]["categories"].keys())
        if args.category == "all":
            categories = site_categories
        elif args.category in site_categories:
            categories = [args.category]
        else:
            log.warning(f"[{site}] Category '{args.category}' is not defined for this site "
                        f"(available: {site_categories}), skipping.")
            continue

        for category in categories:
            for listing_type in types:
                crawl(site, category, listing_type, args.max_pages, inspect=args.inspect,
                      since_days=args.since_days, output_csv=output_csv, stats=stats)
    log.info(f"Run summary: {stats}")
    return stats


if __name__ == "__main__":
    main()
