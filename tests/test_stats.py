from armlab.eval.stats import summarize, wilson


def test_wilson_known_values():
    lo, hi = wilson(95, 100)
    assert abs(lo - 0.8882) < 1e-3 and abs(hi - 0.9785) < 1e-3
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0


def test_summarize():
    s = summarize([True, True, False, True])
    assert s["k"] == 3 and s["n"] == 4 and s["rate"] == 0.75
