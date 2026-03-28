"""測試 src/config.py — 設定載入 + D2 量化參數外部化。"""

from src.config import QuantConfig, Settings, load_settings


class TestLoadSettings:
    def test_missing_file_returns_defaults(self, tmp_path):
        result = load_settings(tmp_path / "nonexistent.yaml")
        assert isinstance(result, Settings)
        assert result.finmind.api_token == ""
        assert result.database.url == "sqlite:///data/stock.db"
        assert result.fetcher.watchlist == ["2330"]

    def test_valid_yaml_loads(self, tmp_path):
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(
            "finmind:\n  api_token: my_token\nfetcher:\n  watchlist:\n    - '2330'\n    - '2317'\n",
            encoding="utf-8",
        )
        result = load_settings(config_file)
        assert result.finmind.api_token == "my_token"
        assert result.fetcher.watchlist == ["2330", "2317"]

    def test_partial_yaml_uses_defaults(self, tmp_path):
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(
            "finmind:\n  api_token: partial_token\n",
            encoding="utf-8",
        )
        result = load_settings(config_file)
        assert result.finmind.api_token == "partial_token"
        # Other fields use defaults
        assert result.fetcher.default_start_date == "2020-01-01"
        assert result.logging.level == "INFO"

    def test_empty_yaml_returns_defaults(self, tmp_path):
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("", encoding="utf-8")
        result = load_settings(config_file)
        assert isinstance(result, Settings)
        assert result.finmind.api_token == ""


# ---------------------------------------------------------------------------
# D2: 量化參數外部化
# ---------------------------------------------------------------------------


class TestQuantConfig:
    def test_defaults(self):
        """未設定 quant 區塊時使用預設值。"""
        s = Settings()
        assert s.quant.trading_cost.commission_rate == 0.001425
        assert s.quant.trading_cost.tax_rate == 0.003
        assert s.quant.atr_multiplier.bull_stop == 1.5
        assert s.quant.score_threshold.bull == 0.45
        assert s.quant.score_threshold.crisis == 0.60

    def test_quant_from_yaml(self, tmp_path):
        """從 YAML 覆蓋量化參數。"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(
            "quant:\n"
            "  trading_cost:\n"
            "    commission_rate: 0.002\n"
            "    tax_rate: 0.005\n"
            "  atr_multiplier:\n"
            "    bull_stop: 2.0\n"
            "  score_threshold:\n"
            "    crisis: 0.70\n",
            encoding="utf-8",
        )
        result = load_settings(config_file)
        assert result.quant.trading_cost.commission_rate == 0.002
        assert result.quant.trading_cost.tax_rate == 0.005
        # 未覆蓋的保持預設
        assert result.quant.trading_cost.slippage_rate == 0.0005
        assert result.quant.atr_multiplier.bull_stop == 2.0
        assert result.quant.atr_multiplier.sideways_stop == 2.0  # 預設
        assert result.quant.score_threshold.crisis == 0.70

    def test_partial_quant(self, tmp_path):
        """只設定部分 quant 子區塊。"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(
            "quant:\n  score_threshold:\n    bear: 0.65\n",
            encoding="utf-8",
        )
        result = load_settings(config_file)
        assert result.quant.score_threshold.bear == 0.65
        # 其他子區塊維持預設
        assert result.quant.trading_cost.commission_rate == 0.001425

    def test_quant_config_standalone(self):
        """QuantConfig 可獨立使用。"""
        qc = QuantConfig()
        assert qc.trading_cost.liquidity_participation_limit == 0.05
        assert qc.atr_multiplier.crisis_target == 1.5
