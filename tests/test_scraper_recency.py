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
    # robots.txt compliance: never request sort/order URLs on PropertyPro.
    assert "recent_sort_query" not in scraper.SITE_CONFIGS["propertypro"]


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


class _TwoPageFetcher:
    """Page 0 has a known listing and a new one; page 1 has one new listing; then nothing."""

    def __init__(self, fresh):
        self.fresh = fresh
        self.detail_calls: list[str] = []
        self.pages = 0
        self.last_status = 200

    def get(self, url):
        if "/property/" in url:
            self.detail_calls.append(url)
            return _soup(f"<h1>Flat</h1><p>&#8358;2,000,000/year Last updated {self.fresh:%Y-%m-%d %H:%M:%S}</p>")
        self.pages += 1
        links = {
            1: ['KNOWN', 'NEWAA'],
            2: ['NEWBB'],
        }.get(self.pages, [])
        return _soup("".join(f'<a href="/property/2-bedroom-flat-apartment-for-rent-yaba-lagos-{c}">x</a>' for c in links))

    @staticmethod
    def polite_sleep():
        pass


def test_known_listings_are_skipped_and_each_page_is_saved(tmp_path):
    now = datetime.now(UTC)
    fake = _TwoPageFetcher(fresh=now - timedelta(days=1))
    known_url = "https://propertypro.ng/property/2-bedroom-flat-apartment-for-rent-yaba-lagos-KNOWN"
    saved_pages: list[list[dict]] = []
    with patch.object(scraper, "Fetcher", return_value=fake):
        stats = scraper.crawl("propertypro", "flat-apartment", "rent", max_pages=5, since_days=7,
                              output_csv=tmp_path / "run.csv", on_page=saved_pages.append,
                              known={known_url: now - timedelta(days=2)})
    # The known listing was never fetched again, but paging carried on past it.
    assert known_url not in fake.detail_calls
    assert stats["already_have"] == 1
    assert stats["kept"] == 2
    # Each page was handed over for saving as soon as it finished.
    assert [len(p) for p in saved_pages] == [1, 1]
    assert saved_pages[0][0]["url"].endswith("NEWAA")


def test_a_failing_page_save_does_not_stop_the_run(tmp_path):
    fake = _TwoPageFetcher(fresh=datetime.now(UTC) - timedelta(days=1))

    def broken(_records):
        raise RuntimeError("database down")

    with patch.object(scraper, "Fetcher", return_value=fake):
        stats = scraper.crawl("propertypro", "flat-apartment", "rent", max_pages=5, since_days=7,
                              output_csv=tmp_path / "run.csv", on_page=broken)
    assert stats["kept"] == 3
    assert stats["page_load_errors"] == 2
    assert len((tmp_path / "run.csv").read_text(encoding="utf-8").splitlines()) == 4  # header + 3, kept for the final pass


def test_partial_counts_already_saved_listings():
    from scraping.runner import run_status

    assert run_status({"blocked_sites": ["npc"], "kept": 0, "already_have": 12}) == "partial"
