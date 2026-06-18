from datetime import date
import unittest

from finance_rag.sources import _recent_filing_candidates, parse_forms


class SourceTests(unittest.TestCase):
    def test_recent_filing_candidates_filters_forms_and_dates(self) -> None:
        recent = {
            "form": ["10-K", "8-K", "10-Q", "10-Q"],
            "filingDate": ["2026-01-31", "2026-02-01", "2025-05-01", "2024-01-01"],
            "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm"],
            "accessionNumber": ["1", "2", "3", "4"],
        }

        candidates = _recent_filing_candidates(
            recent,
            forms={"10-K", "10-Q"},
            start_date=date(2025, 1, 1),
            end_date=date(2026, 12, 31),
        )

        self.assertEqual([candidate["accession_number"] for candidate in candidates], ["1", "3"])

    def test_parse_forms_uppercases_and_strips(self) -> None:
        self.assertEqual(parse_forms("10-k, 10-q,8-k"), {"10-K", "10-Q", "8-K"})


if __name__ == "__main__":
    unittest.main()
