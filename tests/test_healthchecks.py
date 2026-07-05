"""P0 止血包 #5 — healthchecks.io dead-man ping 測試。"""

from __future__ import annotations

import pytest

import src.notification.healthchecks as hc
from src.cli.morning_cmd import _format_crisis_availability


@pytest.fixture()
def _capture_requests(monkeypatch):
    """攔截 request_with_retry，記錄呼叫的 URL。"""
    calls: list[str] = []

    class _Resp:
        status_code = 200

    def _fake_request(method, url, **kwargs):
        calls.append(url)
        return _Resp()

    monkeypatch.setattr(hc, "request_with_retry", _fake_request)
    return calls


def _set_url(monkeypatch, url: str) -> None:
    monkeypatch.setattr(hc.settings.monitoring, "healthchecks_url", url)


class TestPingHealthchecks:
    def test_empty_url_skips_without_request(self, _capture_requests, monkeypatch):
        _set_url(monkeypatch, "")
        assert hc.ping_healthchecks("success") is False
        assert _capture_requests == []

    def test_kind_suffixes(self, _capture_requests, monkeypatch):
        _set_url(monkeypatch, "https://hc-ping.com/abc123")
        assert hc.ping_healthchecks("start") is True
        assert hc.ping_healthchecks("success") is True
        assert hc.ping_healthchecks("fail") is True
        assert _capture_requests == [
            "https://hc-ping.com/abc123/start",
            "https://hc-ping.com/abc123",
            "https://hc-ping.com/abc123/fail",
        ]

    def test_unknown_kind_rejected(self, _capture_requests, monkeypatch):
        _set_url(monkeypatch, "https://hc-ping.com/abc123")
        assert hc.ping_healthchecks("banana") is False
        assert _capture_requests == []

    def test_http_exception_swallowed(self, monkeypatch):
        """網路例外絕不外洩到主流程。"""
        _set_url(monkeypatch, "https://hc-ping.com/abc123")

        def _boom(*a, **kw):
            raise ConnectionError("network down")

        monkeypatch.setattr(hc, "request_with_retry", _boom)
        assert hc.ping_healthchecks("success") is False  # 不 raise

    def test_non_200_returns_false(self, monkeypatch):
        _set_url(monkeypatch, "https://hc-ping.com/abc123")

        class _Resp:
            status_code = 503

        monkeypatch.setattr(hc, "request_with_retry", lambda *a, **kw: _Resp())
        assert hc.ping_healthchecks("success") is False


class TestFormatCrisisAvailability:
    def test_partial_availability_lists_missing(self):
        stress = {
            "availability": {
                "fast_return_5d": True,
                "consec_decline": True,
                "vol_spike": True,
                "panic_volume": True,
                "vix_spike": False,
                "single_day_drop": True,
                "us_vix_spike": True,
            }
        }
        msg = _format_crisis_availability(stress)
        assert msg == "crisis 訊號可用 6/7（不可用：vix_spike）"

    def test_full_availability(self):
        stress = {"availability": {f"s{i}": True for i in range(7)}}
        assert _format_crisis_availability(stress) == "crisis 訊號可用 7/7"

    def test_missing_availability_returns_none(self):
        assert _format_crisis_availability({}) is None
        assert _format_crisis_availability(None) is None
        assert _format_crisis_availability({"signals": {}}) is None
