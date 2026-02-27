"""Discord Webhook 通知模組測試。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.notification.line_notify import format_scan_results, send_message

# ================================================================
# format_scan_results
# ================================================================


class TestFormatScanResults:
    def test_none_returns_default(self):
        result = format_scan_results(None)
        assert "無符合條件" in result

    def test_empty_df_returns_default(self):
        result = format_scan_results(pd.DataFrame())
        assert "無符合條件" in result

    def test_basic_format(self):
        df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "close": [600.0],
                "factor_score": [0.85],
            }
        )
        result = format_scan_results(df)
        assert "2330" in result
        assert "600.0" in result

    def test_top_10_truncation(self):
        df = pd.DataFrame(
            {
                "stock_id": [f"{i:04d}" for i in range(1, 16)],
                "close": [100.0] * 15,
                "factor_score": [0.8] * 15,
            }
        )
        result = format_scan_results(df)
        assert "及其他" in result
        assert "5" in result  # "及其他 5 檔"

    def test_with_rsi_column(self):
        df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "close": [600.0],
                "factor_score": [0.85],
                "rsi_14": [65.0],
            }
        )
        result = format_scan_results(df)
        assert "RSI" in result

    def test_with_foreign_net(self):
        df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "close": [600.0],
                "factor_score": [0.85],
                "foreign_net": [5000.0],
            }
        )
        result = format_scan_results(df)
        assert "外資" in result
        assert "買" in result

    def test_with_yoy_growth(self):
        df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "close": [600.0],
                "factor_score": [0.85],
                "yoy_growth": [15.5],
            }
        )
        result = format_scan_results(df)
        assert "YoY" in result

    def test_exactly_10_no_overflow(self):
        df = pd.DataFrame(
            {
                "stock_id": [f"{i:04d}" for i in range(1, 11)],
                "close": [100.0] * 10,
                "factor_score": [0.8] * 10,
            }
        )
        result = format_scan_results(df)
        assert "及其他" not in result


# ================================================================
# send_message
# ================================================================


class TestSendMessage:
    @patch("src.notification.line_notify.requests.post")
    @patch("src.notification.line_notify.settings")
    def test_success_204(self, mock_settings, mock_post):
        mock_settings.discord.webhook_url = "https://discord.com/api/webhooks/test"
        mock_settings.discord.enabled = True
        mock_settings.discord.username = None
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        mock_post.return_value = mock_resp

        result = send_message("test message")
        assert result is True
        mock_post.assert_called_once()

    @patch("src.notification.line_notify.requests.post")
    @patch("src.notification.line_notify.settings")
    def test_failure_400(self, mock_settings, mock_post):
        mock_settings.discord.webhook_url = "https://discord.com/api/webhooks/test"
        mock_settings.discord.enabled = True
        mock_settings.discord.username = None
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_post.return_value = mock_resp

        result = send_message("test message")
        assert result is False

    @patch("src.notification.line_notify.settings")
    def test_no_url_returns_false(self, mock_settings):
        mock_settings.discord.webhook_url = ""
        result = send_message("test")
        assert result is False

    @patch("src.notification.line_notify.settings")
    def test_disabled_returns_false(self, mock_settings):
        mock_settings.discord.webhook_url = "https://discord.com/api/webhooks/test"
        mock_settings.discord.enabled = False
        result = send_message("test")
        assert result is False
