import unittest

from finance_rag.sources import NEW_YORK_TZ, _canonical_url, _parse_rss_date


class SourceTests(unittest.TestCase):
    def test_parse_rss_date_normalizes_to_utc(self) -> None:
        parsed = _parse_rss_date("Fri, 19 Jun 2026 09:30:00 -0400")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.isoformat(), "2026-06-19T13:30:00+00:00")

    def test_rss_date_maps_to_new_york_calendar_day(self) -> None:
        parsed = _parse_rss_date("Sun, 21 Jun 2026 01:30:00 +0000")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.astimezone(NEW_YORK_TZ).isoformat(), "2026-06-20T21:30:00-04:00")

    def test_canonical_url_removes_query_and_fragment(self) -> None:
        self.assertEqual(
            _canonical_url("https://example.com/story?utm_source=rss#section"),
            "https://example.com/story",
        )


if __name__ == "__main__":
    unittest.main()
