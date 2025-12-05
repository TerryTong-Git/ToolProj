#!/usr/bin/env python3
"""Unit tests for parsers module."""

from src.exps_logistic.parsers import (
    extract_problem_text,
    maybe_strip_fences,
    parse_arithmetic_operands,
    parse_knap_stats,
    parse_lcs_lengths,
    parse_list_of_ints,
    parse_list_of_list_ints,
    parse_rod_N,
    strip_md_code_fences,
)


class TestStripFences:
    """Tests for code fence stripping."""

    def test_strip_md_code_fences_simple(self):
        text = "```python\nprint('hello')\n```"
        result = strip_md_code_fences(text)
        assert result == "print('hello')"

    def test_strip_md_code_fences_empty(self):
        assert strip_md_code_fences("") == ""
        assert strip_md_code_fences(None) == ""

    def test_maybe_strip_fences(self):
        text = "```python\ncode\n```"
        result = maybe_strip_fences(text)
        assert "```" not in result
        assert "python" not in result


class TestParseInts:
    """Tests for integer parsing."""

    def test_parse_list_of_ints(self):
        assert parse_list_of_ints("[1, 2, 3]") == [1, 2, 3]
        assert parse_list_of_ints("1, 2, 3") == [1, 2, 3]
        assert parse_list_of_ints("") == []
        assert parse_list_of_ints("-5, 10, -3") == [-5, 10, -3]

    def test_parse_list_of_list_ints(self):
        result = parse_list_of_list_ints("[[1, 2], [3, 4]]")
        assert result == [[1, 2], [3, 4]]

        result = parse_list_of_list_ints("[]")
        assert result == []


class TestArithmeticParsers:
    """Tests for arithmetic operand parsing."""

    def test_parse_add(self):
        text = "Compute: 123 + 456"
        result = parse_arithmetic_operands("add", text, 3)
        assert result == (123, 456)

    def test_parse_sub(self):
        text = "Compute: 999 - 111"
        result = parse_arithmetic_operands("sub", text, 3)
        assert result == (999, 111)

    def test_parse_mul(self):
        text = "Compute: 12 * 34"
        result = parse_arithmetic_operands("mul", text, 2)
        assert result == (12, 34)

    def test_parse_mix(self):
        text = "Compute: ( 100 + 200 ) * 300"
        result = parse_arithmetic_operands("mix", text, 3)
        assert result == (100, 200)

    def test_parse_no_match(self):
        text = "Some random text"
        result = parse_arithmetic_operands("add", text, 3)
        assert result is None


class TestLCSParser:
    """Tests for LCS length parsing."""

    def test_parse_lcs_lengths(self):
        text = 'S = "abc" T = "defg"'
        Ls, Lt = parse_lcs_lengths(text, 3)
        assert Ls == 3
        assert Lt == 4

    def test_parse_lcs_default(self):
        text = "No strings here"
        Ls, Lt = parse_lcs_lengths(text, 5)
        assert Ls == max(2, 5)
        assert Lt == max(2, 5)


class TestKnapParser:
    """Tests for knapsack parsing."""

    def test_parse_knap_stats(self):
        text = "W = [10, 20, 30] V = [60, 100, 120] C = 50"
        n_items, cap_ratio = parse_knap_stats(text, 3)
        assert n_items == 3
        assert 0 < cap_ratio < 1

    def test_parse_knap_default(self):
        text = "No knapsack data"
        n_items, cap_ratio = parse_knap_stats(text, 5)
        assert n_items == max(3, 5)
        assert cap_ratio == 0.5


class TestRodParser:
    """Tests for rod cutting parsing."""

    def test_parse_rod_N_explicit(self):
        text = "N = 8"
        result = parse_rod_N(text, 5)
        assert result == 8

    def test_parse_rod_N_from_prices(self):
        text = "P = [1, 5, 8, 9, 10, 17, 17, 20]"
        result = parse_rod_N(text, 5)
        assert result == 8


class TestExtractProblemText:
    """Tests for problem text extraction."""

    def test_extract_with_wrapper(self):
        full = "Here is the actual problem: Compute: 1 + 2 Give the solution:"
        result = extract_problem_text(full)
        assert "Compute: 1 + 2" in result

    def test_extract_raw(self):
        full = "Compute: 1 + 2"
        result = extract_problem_text(full)
        assert result == "Compute: 1 + 2"
