from datetime import date, datetime, timezone
import unittest

from finance_rag.prices import parse_sp500_tickers, parse_yahoo_snapshot, to_yahoo_ticker


class PriceTests(unittest.TestCase):
    def test_parse_yahoo_snapshot_uses_latest_minute_and_day_totals(self) -> None:
        timestamps = [
            int(datetime(2026, 6, 22, 13, 30, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 6, 22, 13, 31, tzinfo=timezone.utc).timestamp()),
        ]
        data = {
            "chart": {
                "result": [
                    {
                        "meta": {"chartPreviousClose": 99.0},
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0, 101.0],
                                    "high": [102.0, 104.0],
                                    "low": [98.0, 100.0],
                                    "close": [101.0, 103.0],
                                    "volume": [1000, 1500],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }

        snapshot = parse_yahoo_snapshot(
            data,
            ticker="AAPL",
            snapshot_type="open",
            expected_date=date(2026, 6, 22),
        )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.price, 103.0)
        self.assertEqual(snapshot.day_open, 100.0)
        self.assertEqual(snapshot.day_high, 104.0)
        self.assertEqual(snapshot.day_low, 98.0)
        self.assertEqual(snapshot.volume, 2500)
        self.assertEqual(snapshot.previous_close, 99.0)

    def test_parse_yahoo_snapshot_rejects_previous_trading_day(self) -> None:
        timestamp = int(datetime(2026, 6, 19, 20, 0, tzinfo=timezone.utc).timestamp())
        data = {
            "chart": {
                "result": [
                    {
                        "timestamp": [timestamp],
                        "indicators": {
                            "quote": [
                                {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]}
                            ]
                        },
                    }
                ]
            }
        }

        snapshot = parse_yahoo_snapshot(
            data,
            ticker="AAPL",
            snapshot_type="open",
            expected_date=date(2026, 6, 22),
        )

        self.assertIsNone(snapshot)

    def test_parse_sp500_tickers_and_yahoo_symbol(self) -> None:
        html = """
        <table id="constituents">
          <tr><th>Symbol</th><th>Security</th></tr>
          <tr><td>MMM</td><td>3M</td></tr>
          <tr><td>BRK.B</td><td>Berkshire Hathaway</td></tr>
        </table>
        """

        self.assertEqual(parse_sp500_tickers(html), ["MMM", "BRK.B"])
        self.assertEqual(to_yahoo_ticker("BRK.B"), "BRK-B")


if __name__ == "__main__":
    unittest.main()
