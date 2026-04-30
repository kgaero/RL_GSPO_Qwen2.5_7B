"""Tests for parsing helpers."""

import unittest

from staged_rl.parsing import (
    completion_finished,
    compute_option_letter,
    extract_multichoice_option_letter,
    extract_single_solution_text,
    normalize_numeric_string,
    normalized_exact_match,
    tolerance_match,
)


class ParsingTests(unittest.TestCase):
    def test_normalize_numeric_string(self):
        self.assertEqual(normalize_numeric_string("2.000"), "2")
        self.assertEqual(normalize_numeric_string("0.5000"), "0.5")
        self.assertEqual(normalize_numeric_string("1,200"), "1200")
        self.assertIsNone(normalize_numeric_string("abc"))

    def test_extract_single_solution_text(self):
        text = "<REASONING>work</REASONING><SOLUTION> 12 </SOLUTION>"
        self.assertEqual(extract_single_solution_text(text), "12")

    def test_multichoice_option_letter(self):
        text = "<REASONING>x</REASONING><SOLUTION>The answer is C</SOLUTION>"
        self.assertEqual(extract_multichoice_option_letter(text), "C")
        self.assertEqual(
            extract_multichoice_option_letter("<REASONING>x</REASONING><SOLUTION>The correct answer is (B) 8/11.</SOLUTION>"),
            "B",
        )
        self.assertIsNone(
            extract_multichoice_option_letter("<REASONING>x</REASONING><SOLUTION>Answer: B, not A</SOLUTION>")
        )
        self.assertEqual(compute_option_letter("38", ["28", "38", "52", "62"]), "B")

    def test_exact_and_tolerance(self):
        self.assertTrue(normalized_exact_match("2.0", "2"))
        self.assertTrue(tolerance_match("1.21", "1.2", abs_tol=0.02, rel_tol=0.02))
        self.assertFalse(tolerance_match("1.3", "1.2", abs_tol=0.01, rel_tol=0.01))

    def test_completion_finished(self):
        text = "<REASONING>work</REASONING><SOLUTION>4</SOLUTION>"
        self.assertTrue(completion_finished(text))


if __name__ == "__main__":
    unittest.main()

