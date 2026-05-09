"""factor-diagnostics IC 衰退警示文案測試。

驗證項目：
  1. SSOT — DISCOVERY_KEY_FACTOR_MAP 為單一真相來源，BaseScanner 與 CLI 共用
  2. momentum 關鍵因子為 chip_score（v5：technical 權重歸零後改為 chip_score）
  3. 警示文案正確綁定關鍵因子（避免「momentum 高度依賴 X」這類 stale text）
  4. SSOT — DISCOVERY_IC_HOLDING_DAYS_MAP 與 KEY_FACTOR 對齊：
     fundamental 主導模式用 20 天 holding（防止 5 天視角誤判 inverse）
"""

from __future__ import annotations

from src.cli.discover_cmd import _emit_ic_decay_warning
from src.constants import DISCOVERY_IC_HOLDING_DAYS_MAP, DISCOVERY_KEY_FACTOR_MAP


class TestIcHoldingDaysMapSsot:
    """DISCOVERY_IC_HOLDING_DAYS_MAP（2026-05-09 audit）：
    morning-routine Step 8c 預檢用 mode-aware holding，避免 fundamental 因子被 5 天視角誤判 inverse。"""

    def test_all_modes_have_holding_days(self) -> None:
        """五個模式都應有對應的 IC holding days。"""
        for mode in ("momentum", "swing", "value", "dividend", "growth"):
            assert mode in DISCOVERY_IC_HOLDING_DAYS_MAP
            assert DISCOVERY_IC_HOLDING_DAYS_MAP[mode] >= 1

    def test_chip_factor_uses_short_horizon(self) -> None:
        """chip_score 主導模式（momentum）用短週期（≤10 天）— 資金流訊號短期兌現。"""
        for mode, factor in DISCOVERY_KEY_FACTOR_MAP.items():
            if factor == "chip_score":
                assert DISCOVERY_IC_HOLDING_DAYS_MAP[mode] <= 10, f"{mode}: chip_score 應用短週期 holding"

    def test_fundamental_factor_uses_long_horizon(self) -> None:
        """fundamental_score 主導模式（value/dividend/growth/swing）用 ≥10 天 holding —
        YoY 營收/獲利週期 30+ 天兌現，5 天 holding 視角會誤判 inverse。"""
        for mode, factor in DISCOVERY_KEY_FACTOR_MAP.items():
            if factor == "fundamental_score":
                assert DISCOVERY_IC_HOLDING_DAYS_MAP[mode] >= 10, (
                    f"{mode}: fundamental_score 至少需 10 天 holding，實際 {DISCOVERY_IC_HOLDING_DAYS_MAP[mode]} 天"
                )


class TestStep8cModeAwareHolding:
    """Step 8c (_compute_factor_ic_status) 確實使用 mode-aware holding（SSOT 共用驗證）。"""

    def test_morning_cmd_imports_holding_map(self) -> None:
        """morning_cmd 應 import 並共用同一 DISCOVERY_IC_HOLDING_DAYS_MAP（SSOT）。"""
        from src.cli import morning_cmd

        assert hasattr(morning_cmd, "DISCOVERY_IC_HOLDING_DAYS_MAP"), (
            "morning_cmd 未匯入 DISCOVERY_IC_HOLDING_DAYS_MAP，Step 8c 無法 mode-aware"
        )
        assert morning_cmd.DISCOVERY_IC_HOLDING_DAYS_MAP is DISCOVERY_IC_HOLDING_DAYS_MAP, (
            "morning_cmd 應與 constants 共用同一物件（SSOT）"
        )

    def test_all_key_factor_modes_have_holding(self) -> None:
        """每個 KEY_FACTOR_MAP 的模式都需有對應的 holding_days，避免 KeyError 漏網。"""
        for mode in DISCOVERY_KEY_FACTOR_MAP:
            assert mode in DISCOVERY_IC_HOLDING_DAYS_MAP, f"{mode} 缺少 holding_days 設定"


class TestKeyFactorMapSsot:
    def test_market_scanner_shares_same_object(self) -> None:
        """MarketScanner._KEY_FACTOR_MAP 與 constants 應為同一物件（SSOT）。"""
        from src.discovery.scanner._base import MarketScanner

        assert MarketScanner._KEY_FACTOR_MAP is DISCOVERY_KEY_FACTOR_MAP

    def test_momentum_key_is_chip_score(self) -> None:
        """v5（2026-05-09 audit）：technical 權重歸零後，chip_score（0.55）成為最高權重維度，
        故 momentum 關鍵因子由 technical_score 改為 chip_score。"""
        assert DISCOVERY_KEY_FACTOR_MAP["momentum"] == "chip_score"

    def test_all_modes_have_mapping(self) -> None:
        """五個模式都應有對應的關鍵因子。"""
        for mode in ("momentum", "swing", "value", "dividend", "growth"):
            assert mode in DISCOVERY_KEY_FACTOR_MAP
            assert DISCOVERY_KEY_FACTOR_MAP[mode].endswith("_score")

    def test_key_factor_is_max_weight_dimension_in_bull(self) -> None:
        """KEY_FACTOR 應為各模式 bull regime 中最大權重的維度（避免誤判）。

        防止 swing 之前的 bug：fundamental 0.40 是最大權重，但 KEY_FACTOR
        錯設為 chip_score (0.20，第三大)，導致 IC 預檢用第三大維度判斷停用。
        """
        from src.regime.detector import REGIME_WEIGHTS

        for mode, key_factor_score in DISCOVERY_KEY_FACTOR_MAP.items():
            weights = REGIME_WEIGHTS[mode]["bull"]
            max_dim = max(weights, key=weights.get)
            expected_key = f"{max_dim}_score"
            assert key_factor_score == expected_key, (
                f"{mode}: KEY_FACTOR={key_factor_score} 應為 bull 最大權重維度 "
                f"{expected_key}（權重 {weights[max_dim]:.2f}），實際權重 "
                f"{weights[key_factor_score.replace('_score', '')]:.2f}"
            )


class TestEmitIcDecayWarning:
    def test_non_key_factor_no_dependency_message(self, capsys) -> None:
        """非關鍵因子衰退不應觸發『高度依賴』訊息。"""
        # v5：momentum 關鍵因子為 chip_score，news_score / technical_score 不應觸發強警示
        _emit_ic_decay_warning(mode="momentum", factor="news_score", latest_ic=-0.05)
        out = capsys.readouterr().out
        assert "高度依賴" not in out
        assert "★關鍵因子" not in out
        assert "news_score IC 衰退至 -0.0500" in out  # 但基本警示仍應出現

    def test_key_factor_emits_dependency_message(self, capsys) -> None:
        """關鍵因子衰退應強警示（v5：momentum 關鍵因子改為 chip_score）。"""
        _emit_ic_decay_warning(mode="momentum", factor="chip_score", latest_ic=-0.05)
        out = capsys.readouterr().out
        assert "★關鍵因子" in out
        assert "momentum 模式高度依賴 chip_score" in out

    def test_swing_key_factor_fundamental_score(self, capsys) -> None:
        """swing 關鍵因子為 fundamental_score（bull regime 0.40，最大權重維度）。

        歷史 bug：曾誤設為 chip_score（0.20，第三大），造成 Step 8c IC 預檢
        以 chip IC 反向誤殺 swing 模式停用。詳見 src/constants.py 註解。
        """
        _emit_ic_decay_warning(mode="swing", factor="fundamental_score", latest_ic=0.02)
        out = capsys.readouterr().out
        assert "swing 模式高度依賴 fundamental_score" in out

    def test_value_key_factor_fundamental(self, capsys) -> None:
        """value/dividend/growth 關鍵因子皆為 fundamental_score。"""
        _emit_ic_decay_warning(mode="value", factor="fundamental_score", latest_ic=0.05)
        out = capsys.readouterr().out
        assert "value 模式高度依賴 fundamental_score" in out

    def test_unknown_mode_falls_back_silent(self, capsys) -> None:
        """未知模式不觸發強警示但仍印基本訊息。"""
        _emit_ic_decay_warning(mode="unknown_mode", factor="technical_score", latest_ic=-0.10)
        out = capsys.readouterr().out
        assert "高度依賴" not in out
        assert "technical_score IC 衰退至 -0.1000" in out

    def test_threshold_displayed(self, capsys) -> None:
        """門檻值應顯示在警示文案內，便於使用者理解觸發條件。"""
        _emit_ic_decay_warning(mode="momentum", factor="news_score", latest_ic=0.05, ic_threshold=0.10)
        out = capsys.readouterr().out
        assert "< 0.10 門檻" in out


class TestIcActionsPropagation:
    """IC-aware actions 透傳到 DiscoveryResult（供 CLI 表格標記欄位狀態）。"""

    def test_discovery_result_has_ic_actions_field(self) -> None:
        """DiscoveryResult 應有 ic_actions 預設空 dict。"""
        import pandas as pd

        from src.discovery.scanner._functions import DiscoveryResult

        result = DiscoveryResult(rankings=pd.DataFrame(), total_stocks=0, after_coarse=0)
        assert hasattr(result, "ic_actions")
        assert result.ic_actions == {}

    def test_discovery_result_accepts_ic_actions(self) -> None:
        """ic_actions 可直接於建構時傳入。"""
        import pandas as pd

        from src.discovery.scanner._functions import DiscoveryResult

        actions = {"technical_score": "neutralized", "news_score": "flipped"}
        result = DiscoveryResult(rankings=pd.DataFrame(), total_stocks=0, after_coarse=0, ic_actions=actions)
        assert result.ic_actions == actions

    def test_scanner_init_creates_empty_ic_actions(self) -> None:
        """MarketScanner 初始化時 _ic_actions 為空 dict。"""
        from src.discovery.scanner._momentum import MomentumScanner

        scanner = MomentumScanner(use_ic_adjustment=False)
        assert hasattr(scanner, "_ic_actions")
        assert scanner._ic_actions == {}


class TestTopTableMarking:
    """CLI Top N 表格 (N)/(F) 欄位標記與圖例。"""

    def _make_result(self, ic_actions: dict[str, str]):
        """組裝最小可用 DiscoveryResult 供 cmd_discover 渲染。"""
        import pandas as pd

        from src.discovery.scanner._functions import DiscoveryResult

        rankings = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "stock_id": "1234",
                    "stock_name": "TestCo",
                    "close": 100.0,
                    "composite_score": 0.65,
                    "technical_score": 0.5,
                    "chip_score": 0.5,
                    "fundamental_score": 0.6,
                    "chip_tier": "5F",
                    "industry_category": "半導體業",
                    "sector_bonus": 0.03,
                }
            ]
        )
        return DiscoveryResult(
            rankings=rankings, total_stocks=100, after_coarse=20, mode="momentum", ic_actions=ic_actions
        )

    def test_marks_neutralized_columns_in_header(self, capsys, monkeypatch) -> None:
        """中性化欄位應在表頭加 (N) 標記 + 圖例。"""
        import argparse

        from src.cli import discover_cmd

        # 用 monkeypatch 攔截 scanner.run() 回傳預先組好的 result
        result = self._make_result({"technical_score": "neutralized", "chip_score": "neutralized"})

        class _StubScanner:
            def __init__(self, *args, **kwargs): ...
            def run(self):
                return result

        monkeypatch.setattr(discover_cmd, "init_db", lambda: None)
        monkeypatch.setattr(discover_cmd, "ensure_sync_market_data", lambda *a, **kw: None)
        monkeypatch.setattr(discover_cmd, "_save_discovery_records", lambda *a, **kw: None)
        from src.discovery import scanner as scanner_pkg

        monkeypatch.setattr(scanner_pkg, "MomentumScanner", _StubScanner)

        args = argparse.Namespace(
            mode="momentum",
            min_price=10,
            max_price=2000,
            min_volume=1000,
            top=20,
            sync_days=25,
            weekly_confirm=False,
            use_ic_adjustment=False,
            compare=False,
            verbose=False,
            export=None,
            notify=False,
        )
        discover_cmd.cmd_discover(args)
        out = capsys.readouterr().out

        assert "技術(N)" in out
        assert "籌碼(N)" in out
        assert "基本" in out and "基本(N)" not in out  # 未中性化的不加標記
        assert "(N)=IC<0.05 中性化" in out  # 圖例

    def test_marks_flipped_columns(self, capsys, monkeypatch) -> None:
        """反向欄位應在表頭加 (F) 標記。"""
        import argparse

        from src.cli import discover_cmd

        result = self._make_result({"technical_score": "flipped"})

        class _StubScanner:
            def __init__(self, *args, **kwargs): ...
            def run(self):
                return result

        monkeypatch.setattr(discover_cmd, "init_db", lambda: None)
        monkeypatch.setattr(discover_cmd, "ensure_sync_market_data", lambda *a, **kw: None)
        monkeypatch.setattr(discover_cmd, "_save_discovery_records", lambda *a, **kw: None)
        from src.discovery import scanner as scanner_pkg

        monkeypatch.setattr(scanner_pkg, "MomentumScanner", _StubScanner)

        args = argparse.Namespace(
            mode="momentum",
            min_price=10,
            max_price=2000,
            min_volume=1000,
            top=20,
            sync_days=25,
            weekly_confirm=False,
            use_ic_adjustment=False,
            compare=False,
            verbose=False,
            export=None,
            notify=False,
        )
        discover_cmd.cmd_discover(args)
        out = capsys.readouterr().out

        assert "技術(F)" in out
        assert "(F)=IC 反向已翻轉" in out

    def test_no_legend_when_all_kept(self, capsys, monkeypatch) -> None:
        """全部 kept 時不應顯示圖例（避免雜訊）。"""
        import argparse

        from src.cli import discover_cmd

        result = self._make_result({"technical_score": "kept", "chip_score": "kept"})

        class _StubScanner:
            def __init__(self, *args, **kwargs): ...
            def run(self):
                return result

        monkeypatch.setattr(discover_cmd, "init_db", lambda: None)
        monkeypatch.setattr(discover_cmd, "ensure_sync_market_data", lambda *a, **kw: None)
        monkeypatch.setattr(discover_cmd, "_save_discovery_records", lambda *a, **kw: None)
        from src.discovery import scanner as scanner_pkg

        monkeypatch.setattr(scanner_pkg, "MomentumScanner", _StubScanner)

        args = argparse.Namespace(
            mode="momentum",
            min_price=10,
            max_price=2000,
            min_volume=1000,
            top=20,
            sync_days=25,
            weekly_confirm=False,
            use_ic_adjustment=False,
            compare=False,
            verbose=False,
            export=None,
            notify=False,
        )
        discover_cmd.cmd_discover(args)
        out = capsys.readouterr().out

        assert "(N)" not in out and "(F)" not in out
        assert "圖例" not in out

    def test_legend_only_for_visible_columns(self, capsys, monkeypatch) -> None:
        """僅 news_score 被翻轉、表格不顯示 news 欄時，不應誤顯示 (F) 圖例。"""
        import argparse

        from src.cli import discover_cmd

        # news_score flipped，但表格只顯示 technical/chip/fundamental → 圖例應為空
        result = self._make_result({"news_score": "flipped"})

        class _StubScanner:
            def __init__(self, *args, **kwargs): ...

            def run(self):
                return result

        monkeypatch.setattr(discover_cmd, "init_db", lambda: None)
        monkeypatch.setattr(discover_cmd, "ensure_sync_market_data", lambda *a, **kw: None)
        monkeypatch.setattr(discover_cmd, "_save_discovery_records", lambda *a, **kw: None)
        from src.discovery import scanner as scanner_pkg

        monkeypatch.setattr(scanner_pkg, "MomentumScanner", _StubScanner)

        args = argparse.Namespace(
            mode="momentum",
            min_price=10,
            max_price=2000,
            min_volume=1000,
            top=20,
            sync_days=25,
            weekly_confirm=False,
            use_ic_adjustment=False,
            compare=False,
            verbose=False,
            export=None,
            notify=False,
        )
        discover_cmd.cmd_discover(args)
        out = capsys.readouterr().out

        # news 欄不在表格中 → 不應顯示 (F) 圖例
        assert "圖例" not in out
        assert "(F)=IC 反向已翻轉" not in out
