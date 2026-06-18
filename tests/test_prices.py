from datetime import date
import unittest

from finance_rag.prices import (
    estimate_daily_price_rows,
    estimate_daily_price_storage_mb,
    parse_sp500_tickers,
    parse_yahoo_chart_response,
    to_yahoo_ticker,
    years_before,
)


class PriceTests(unittest.TestCase):
    def test_parse_yahoo_chart_response(self) -> None:
        data = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1704067200, 1704153600],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0, 101.0],
                                    "high": [105.0, 106.0],
                                    "low": [99.0, 100.5],
                                    "close": [104.0, 103.0],
                                    "volume": [1000, 1200],
                                }
                            ],
                            "adjclose": [{"adjclose": [103.5, 102.5]}],
                        },
                    }
                ],
                "error": None,
            }
        }

        bars = parse_yahoo_chart_response(data, ticker="AAPL")

        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].ticker, "AAPL")
        self.assertEqual(bars[0].trade_date, date(2024, 1, 1))
        self.assertEqual(bars[1].adjusted_close, 102.5)

    def test_parse_sp500_tickers(self) -> None:
        html = """
        <table id="constituents">
          <tr><th>Symbol</th><th>Security</th></tr>
          <tr><td>MMM</td><td>3M</td></tr>
          <tr><td>BRK.B</td><td>Berkshire Hathaway</td></tr>
        </table>
        """

        self.assertEqual(parse_sp500_tickers(html), ["MMM", "BRK.B"])

    def test_yahoo_ticker_replaces_dot_classes(self) -> None:
        self.assertEqual(to_yahoo_ticker("BRK.B"), "BRK-B")

    def test_storage_estimate_for_sp500_10_years(self) -> None:
        rows = estimate_daily_price_rows(ticker_count=500, years=10)
        low, high = estimate_daily_price_storage_mb(ticker_count=500, years=10)

        self.assertEqual(rows, 1_260_000)
        self.assertGreater(low, 0)
        self.assertGreater(high, low)

    def test_years_before_handles_leap_day(self) -> None:
        self.assertEqual(years_before(date(2024, 2, 29), 1), date(2023, 2, 28))


if __name__ == "__main__":
    unittest.main()
