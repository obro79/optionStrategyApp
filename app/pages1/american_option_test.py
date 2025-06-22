import pytest
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Import your module here. Adjust the import path as needed.
from american_option import AmericanOption


def test_constructor_sets_attributes_correctly():
    bs = AmericanOption(10, 5, 1, 0.2, 0.05, 5, option_type="call")
    expected = (10, 5, 1, 0.2, 0.05, 5, "call")
    actual = (
        bs.spot_price,
        bs.strike_price,
        bs.time_to_maturity,
        bs.volatility,
        bs.interest_rate,
        bs.num_steps,
        bs.option_type,
    )
    assert actual == expected


def test_price_one_step_call():
    # Manual calculation for a one-step binomial tree with zero interest
    spot, strike, ttm, vol, r, steps = 100, 100, 1, 0.1, 0.0, 1
    opt = AmericanOption(spot, strike, ttm, vol, r, steps, option_type="call")
    price = opt.price()

    # Compute expected analytically for this case
    dt = ttm / steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    intrinsic_up = max(0, spot * u - strike)
    expected = p * intrinsic_up

    assert price == pytest.approx(expected, rel=1e-6)


def test_price_monotonic_in_volatility():
    # A call option price should not decrease when volatility increases
    opt_low = AmericanOption(100, 100, 1, 0.1, 0.05, 50, option_type="call")
    opt_high = AmericanOption(100, 100, 1, 0.2, 0.05, 50, option_type="call")
    price_low = opt_low.price()
    price_high = opt_high.price()
    assert price_high >= price_low


def test_invalid_option_type_behaves_as_put():
    # Any unrecognized option_type should default to put logic
    params = (100, 100, 1, 0.2, 0.05, 10)
    opt_invalid = AmericanOption(*params, option_type="invalid")
    opt_put = AmericanOption(*params, option_type="put")
    assert opt_invalid.price() == pytest.approx(opt_put.price())


def test_visualize_binomial_tree(monkeypatch):
    # Ensure visualize_binomial_tree calls streamlit.pyplot exactly once with a figure
    calls = []
    def fake_pyplot(fig):
        calls.append(fig)

    monkeypatch.setattr(st, 'pyplot', fake_pyplot)
    opt = AmericanOption(100, 100, 1, 0.2, 0.05, 3, option_type="call")
    opt.visualize_binomial_tree()

    assert len(calls) == 1
    assert isinstance(calls[0], plt.Figure)
