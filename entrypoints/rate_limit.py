"""Usage limits for the requests that spend AI credit (Groq, and Tavily for advice).

Three layers, checked together before any AI call:
- per device (the anonymous X-Client-Id the app sends): stops one browser hammering the chat
- per network address: catches scripts that drop or rotate device ids. It is looser, because
  many phones on Nigerian mobile networks share one public address
- a daily total for the whole app: a hard ceiling on spend that no header can get around

Counts live in memory. The API runs as one Render instance, so no shared store is needed,
and a restart only ever loosens the limits. Every number can be changed with an env var.
"""
import hashlib
import os
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta

LAGOS = timedelta(hours=1)  # West Africa Time, no daylight saving
MAX_ALERTS_PER_DAY = 50


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class Limited(Exception):
    def __init__(self, scope: str, message: str, retry_after: int):
        super().__init__(message)
        self.scope = scope
        self.message = message
        self.retry_after = max(1, int(retry_after))


class UsageLimiter:
    def __init__(self, device_minute: int, device_day: int, ip_minute: int, ip_day: int, global_day: int,
                 clock=time.time):
        self.limits = {"device_minute": device_minute, "device_day": device_day,
                       "ip_minute": ip_minute, "ip_day": ip_day, "global_day": global_day}
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: dict[str, deque] = {}  # key -> timestamps in the last minute
        self._daily: dict[str, int] = {}  # key -> count for the current Lagos day
        self._day: str | None = None
        self._alerted: set[str] = set()

    @classmethod
    def from_env(cls) -> "UsageLimiter":
        return cls(
            device_minute=_env_int("LIMIT_DEVICE_PER_MINUTE", 8),
            device_day=_env_int("LIMIT_DEVICE_PER_DAY", 100),
            ip_minute=_env_int("LIMIT_IP_PER_MINUTE", 20),
            ip_day=_env_int("LIMIT_IP_PER_DAY", 400),
            global_day=_env_int("LIMIT_GLOBAL_PER_DAY", 3000),
        )

    # -- time helpers -------------------------------------------------------------------------
    def _lagos_now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), UTC) + LAGOS

    def _seconds_to_midnight(self) -> int:
        now = self._lagos_now()
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return int((midnight - now).total_seconds()) + 1

    def _roll_day(self) -> None:
        today = self._lagos_now().date().isoformat()
        if today != self._day:
            self._day = today
            self._daily.clear()
            self._alerted.clear()

    def _minute_window(self, key: str, now: float) -> deque:
        q = self._recent.setdefault(key, deque())
        while q and q[0] <= now - 60:
            q.popleft()
        return q

    def _tidy(self, now: float) -> None:
        if len(self._recent) > 20_000:
            for key in [k for k, q in self._recent.items() if not q or q[-1] <= now - 60]:
                del self._recent[key]

    # -- the check ----------------------------------------------------------------------------
    def check(self, device: str | None, ip: str | None) -> None:
        """Count one AI request, or raise Limited without counting it."""
        with self._lock:
            now = self._clock()
            self._roll_day()
            self._tidy(now)
            keys = [("ip", f"ip:{ip}")] if ip else []
            if device:
                keys.insert(0, ("device", f"dev:{device}"))

            if self._daily.get("global", 0) >= self.limits["global_day"]:
                raise Limited("global_day",
                              "Adugbo has answered a lot of questions today and is taking a break until midnight "
                              "(Lagos time). Area guides, Compare and Move-in cost still work.",
                              self._seconds_to_midnight())
            for kind, key in keys:
                if self._daily.get(key, 0) >= self.limits[f"{kind}_day"]:
                    raise Limited(f"{kind}_day",
                                  "You've reached today's limit for questions. It resets at midnight (Lagos time). "
                                  "Area guides, Compare and Move-in cost still work.",
                                  self._seconds_to_midnight())
            for kind, key in keys:
                q = self._minute_window(key, now)
                if len(q) >= self.limits[f"{kind}_minute"]:
                    wait = int(q[0] + 60 - now) + 1 if q else 60  # a limit of 0 blocks with an empty window
                    raise Limited(f"{kind}_minute",
                                  f"You're asking faster than Adugbo can keep up. Wait about {wait} seconds and try again.",
                                  wait)

            self._daily["global"] = self._daily.get("global", 0) + 1
            for _kind, key in keys:
                self._daily[key] = self._daily.get(key, 0) + 1
                self._minute_window(key, now).append(now)

    def should_alert(self, scope: str, who: str) -> bool:
        """True the first time a given limit is hit by a given caller today, capped per day."""
        with self._lock:
            self._roll_day()
            key = f"{scope}:{who}"
            if key in self._alerted or len(self._alerted) >= MAX_ALERTS_PER_DAY:
                return False
            self._alerted.add(key)
            return True

    def usage(self) -> dict:
        with self._lock:
            self._roll_day()
            return {"day": self._day, "questions_today": self._daily.get("global", 0), "limits": dict(self.limits)}


def masked(value: str | None) -> str:
    """A short, non-reversible label for logs and alerts, so raw addresses are never stored."""
    if not value:
        return "unknown"
    return hashlib.sha256(value.encode()).hexdigest()[:10]


def device_id(header: str | None) -> str | None:
    return header if header and len(header) <= 64 else None


limiter = UsageLimiter.from_env()
