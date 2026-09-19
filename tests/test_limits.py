from app.limits import DailyBudget, SlidingWindowLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_allows_up_to_the_limit_then_blocks():
    limiter = SlidingWindowLimiter(limit=3, window_seconds=3600, clock=FakeClock())
    assert [limiter.check("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = limiter.check("a")
    assert allowed is False
    assert retry_after > 0


def test_retry_after_says_when_the_oldest_hit_expires():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=2, window_seconds=3600, clock=clock)
    limiter.check("a")
    clock.now += 600
    limiter.check("a")
    clock.now += 60
    _, retry_after = limiter.check("a")
    assert 2900 <= retry_after <= 3000  # about 3600 - 660 seconds


def test_slots_free_up_after_the_window():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=1, window_seconds=60, clock=clock)
    assert limiter.check("a")[0] is True
    assert limiter.check("a")[0] is False
    clock.now += 61
    assert limiter.check("a")[0] is True


def test_visitors_are_counted_separately():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=3600, clock=FakeClock())
    assert limiter.check("a")[0] is True
    assert limiter.check("b")[0] is True
    assert limiter.check("a")[0] is False


def test_a_blocked_attempt_does_not_extend_the_block():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=1, window_seconds=100, clock=clock)
    limiter.check("a")
    for _ in range(5):
        clock.now += 10
        assert limiter.check("a")[0] is False
    clock.now += 51  # 101 s after the only recorded hit
    assert limiter.check("a")[0] is True


def test_daily_budget_runs_out_and_reports_what_is_left():
    budget = DailyBudget(limit=2, today=lambda: "2026-09-19")
    assert budget.remaining() == 2
    assert budget.try_spend() is True
    assert budget.try_spend() is True
    assert budget.try_spend() is False
    assert budget.remaining() == 0


def test_daily_budget_resets_on_a_new_day():
    day = {"value": "2026-09-19"}
    budget = DailyBudget(limit=1, today=lambda: day["value"])
    assert budget.try_spend() is True
    assert budget.try_spend() is False
    day["value"] = "2026-09-20"
    assert budget.remaining() == 1
    assert budget.try_spend() is True
