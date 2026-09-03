"""How the weekly regression picks its area and window.

Not marked network: the seeding is what needs checking here, not the archive.
"""

from tests.test_live_products import week_seed


def test_an_empty_seed_is_not_a_seed(monkeypatch):
    """CI sets the variable to "" when no seed is given, and every scheduled run does the
    same. Taken as a seed, every run would pick the same area forever."""
    monkeypatch.setenv("OPERA_FETCH_SEED", "")
    assert week_seed().startswith("20"), "an empty seed must fall back to the ISO week"


def test_a_named_seed_is_used_as_given(monkeypatch):
    monkeypatch.setenv("OPERA_FETCH_SEED", "2026-W12")
    assert week_seed() == "2026-W12"


def test_the_week_seed_moves_on_but_holds_within_a_week(monkeypatch):
    """A fixed area passes forever on one lucky corner; a free one cannot be replayed."""
    import random

    monkeypatch.delenv("OPERA_FETCH_SEED", raising=False)
    seed = week_seed()
    assert seed == week_seed(), "two runs in one week must agree"
    assert random.Random(seed).random() != random.Random("2026-W01").random()
