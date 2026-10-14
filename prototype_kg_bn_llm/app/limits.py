"""Protection for a public deployment (a value of 0 switches that limit off): per-visitor and daily caps on LLM calls (they spend the owner's credit) and a cap on active sessions (CPU)."""
import os
import time
from collections import defaultdict, deque


class LLMLimiter:
    def __init__(self, per_ip_hour=None, per_day=None):
        self.per_ip_hour = int(os.environ.get("LLM_PER_IP_HOUR", 6)) if per_ip_hour is None else per_ip_hour
        self.per_day = int(os.environ.get("LLM_PER_DAY", 150)) if per_day is None else per_day
        self.ip = defaultdict(deque); self.day = (time.strftime("%Y-%m-%d"), 0)

    def allow(self, ip, now=None):
        """True and the call is counted, or (False, reason)."""
        now = time.time() if now is None else now; d = time.strftime("%Y-%m-%d", time.gmtime(now))
        if self.day[0] != d: self.day = (d, 0)
        q = self.ip[ip]
        while q and now - q[0] > 3600: q.popleft()
        if self.per_day > 0 and self.day[1] >= self.per_day: return False, f"the public demo reached its daily limit of {self.per_day} LLM calls; try again tomorrow or use the rule-based text"
        if self.per_ip_hour > 0 and len(q) >= self.per_ip_hour: return False, f"you reached the limit of {self.per_ip_hour} LLM calls per hour on this public demo; use the rule-based text meanwhile"
        q.append(now); self.day = (d, self.day[1] + 1); return True, ""


MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", 6)); IDLE_S = int(os.environ.get("SESSION_IDLE_S", 600))
