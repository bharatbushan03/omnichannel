from ingestion.rate_limit import SlidingWindowRateLimiter


def test_window_blocks_then_expires():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_s=60)
    assert limiter.allow("10.0.0.1", now=10)
    assert limiter.allow("10.0.0.1", now=11)
    assert limiter.allow("10.0.0.1", now=12) is False
    assert limiter.allow("10.0.0.2", now=12) is True
    assert limiter.allow("10.0.0.1", now=71) is True


def test_independent_keys():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_s=10)
    assert limiter.allow("a", now=1)
    assert limiter.allow("b", now=1)
    assert limiter.allow("a", now=2) is False
