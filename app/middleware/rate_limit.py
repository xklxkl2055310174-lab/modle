import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class Bucket:
    tokens: float
    updated: float

    def refill(self, now, qps, burst):
        self.tokens = min(burst, self.tokens + (now - self.updated) * qps)
        self.updated = now


class RateLimiter:
    """仅由单个事件循环同步访问，不 await，不需要锁；单实例限流。"""

    def __init__(self, settings, clock=time.monotonic):
        self.settings, self.clock = settings, clock
        self.global_bucket = Bucket(settings.rate_limit_burst, clock())
        self.keys = OrderedDict()

    def allow(self, key: str) -> bool:
        s, now = self.settings, self.clock()
        checks = []
        if s.rate_limit_qps:
            self.global_bucket.refill(now, s.rate_limit_qps, s.rate_limit_burst)
            checks.append(self.global_bucket)
        if s.key_rate_limit_qps:
            while (
                self.keys and next(iter(self.keys.values())).updated < now - s.rate_limit_key_ttl_s
            ):
                self.keys.popitem(last=False)
            if key not in self.keys:
                if len(self.keys) >= s.rate_limit_max_keys:
                    self.keys.popitem(last=False)
                self.keys[key] = Bucket(s.key_rate_limit_burst, now)
            bucket = self.keys[key]
            self.keys.move_to_end(key)
            bucket.refill(now, s.key_rate_limit_qps, s.key_rate_limit_burst)
            checks.append(bucket)
        if any(bucket.tokens < 1 for bucket in checks):
            return False
        for bucket in checks:
            bucket.tokens -= 1
        return True
