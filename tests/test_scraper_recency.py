from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from bs4 import BeautifulSoup

from scraping import scraper


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def test_npc_listed_at_from_json_ld():
    soup = _soup('<script type="application/ld+json">{"datePosted":"2026-11-17T15:58:04+01:00"}</script>')
    assert scraper._parse_npc_listed_at(soup).isoformat() == "2026-11-17T15:58:04+01:00"


def test_npc_listed_at_from_visible_text():
    soup = _soup("<div>Ref ID: #1 Posted 17 Nov 2026</div>")
    assert scraper._parse_npc_listed_at(soup).date().isoformat() == "2026-11-17"


def test_propertypro_listed_at():
    soup = _soup("<p>Lagos Last updated 2026-11-17 13:02:15.0 View</p>")
    assert scraper._parse_propertypro_listed_at(soup).isoformat() == "2026-11-17T13:02:15+01:00"


def test_missing_date_returns_none():
    assert scraper._parse_npc_listed_at(_soup("<p>nothing</p>")) is None
    assert scraper._parse_propertypro_listed_at(_soup("<p>nothing</p>")) is None


class _FakeFetcher:
    """Serves one search page with two listings (one fresh, one old), then an empty page."""

    def __init__(self, fresh, old):
        self.pages = 0
        self.fresh, self.old = fresh, old

    def get(self, url):
        if "/property/" in url:
            when = self.fresh if url.endswith("AAAAA") else self.old
            return _soup(f"<h1>Flat</h1><p>&#8358;1,500,000/year Last updated {when:%Y-%m-%d %H:%M:%S}</p>")
        self.pages += 1
        if self.pages > 1:
            return _soup("<p>no links</p>")
        return _soup(
            '<a href="/property/2-bedroom-flat-apartment-for-rent-yaba-lagos-AAAAA">a</a>'
            '<a href="/property/2-bedroom-flat-apartment-for-rent-yaba-lagos-BBBBB">b</a>'
        )

    @staticmethod
    def polite_sleep():
        pass


def test_crawl_keeps_only_recent_listings(tmp_path):
    now = datetime.now(UTC)
    fake = _FakeFetcher(fresh=now - timedelta(days=1), old=now - timedelta(days=30))
    out = tmp_path / "run.csv"
    with patch.object(scraper, "Fetcher", return_value=fake):
        stats = scraper.crawl("propertypro", "flat-apartment", "rent", max_pages=3, since_days=7, output_csv=out)
    assert stats["discovered"] == 2
    assert stats["kept"] == 1
    assert stats["too_old"] == 1
    rows = out.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2  # header + one fresh listing
    assert "AAAAA" in rows[1]
    assert "sort=postedOn" in str(scraper.SITE_CONFIGS["propertypro"]["recent_sort_query"])


class _BlockedFetcher:
    """Every request is refused with HTTP 403, like a site blocking cloud servers."""

    def __init__(self):
        self.calls = 0
        self.last_status = None

    def get(self, url):
        self.calls += 1
        self.last_status = 403
        return None

    @staticmethod
    def polite_sleep():
        pass


def test_blocked_site_is_skipped_after_two_pages(tmp_path):
    fake = _BlockedFetcher()
    with patch.object(scraper, "Fetcher", return_value=fake):
        stats = scraper.main(["--site", "npc", "--category", "all", "--type", "rent", "--max-pages", "30",
                              "--since-days", "7", "--output", str(tmp_path / "run.csv")])
    assert stats["blocked_sites"] == ["npc"]
    assert fake.calls == scraper.BLOCKED_AFTER_PAGES  # second category never fetched
    assert stats["kept"] == 0


def test_run_status_reflects_blocked_sites():
    from scraping.runner import run_status

    assert run_status({"blocked_sites": [], "kept": 0}) == "success"
    assert run_status({"blocked_sites": ["npc"], "kept": 40}) == "partial"
    assert run_status({"blocked_sites": ["npc", "propertypro"], "kept": 0}) == "failed"
