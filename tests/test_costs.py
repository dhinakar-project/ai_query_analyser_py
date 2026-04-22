"""Unit tests for cost tracking module."""
import pytest
from observability.costs import calculate_cost, format_cost, reset_session_cost, get_session_cost, add_cost


def test_calculate_cost_zero_tokens():
    cost = calculate_cost(0, 0)
    assert cost == 0.0


def test_calculate_cost_positive():
    cost = calculate_cost(1000, 500)
    assert cost > 0


def test_format_cost():
    formatted = format_cost(0.00234)
    assert formatted.startswith("$")
    assert "0.002" in formatted


def test_session_cost_accumulates():
    reset_session_cost()
    add_cost(0.001)
    add_cost(0.002)
    total = get_session_cost()
    assert abs(total - 0.003) < 1e-9


def test_reset_session_cost():
    add_cost(0.999)
    reset_session_cost()
    assert get_session_cost() == 0.0
