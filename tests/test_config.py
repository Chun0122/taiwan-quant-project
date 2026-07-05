"""測試 src/config.py — 設定載入 + D2 量化參數外部化 + A5 拆檔（雙檔模式）。"""

from src.config import QuantConfig, Settings, _deep_merge, load_settings


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


class TestRotationCostPerMode:
    """per-mode 成本閘門覆蓋（投研 2026-06-07：Gate B 對 momentum/swing 反向）。"""

    def test_for_mode_no_override_returns_global(self):
        from src.config import RotationCostConfig

        rc = RotationCostConfig(enabled=True, min_hold_days=7, score_gap_threshold=0.05, weekly_swap_cap=4)
        m = rc.for_mode("momentum")
        assert (m.enabled, m.min_hold_days, m.score_gap_threshold, m.weekly_swap_cap) == (True, 7, 0.05, 4)
        # 無覆蓋時直接回傳同一物件
        assert rc.for_mode("momentum") is rc
        assert rc.for_mode(None) is rc

    def test_for_mode_partial_override_inherits_rest(self):
        from src.config import RotationCostConfig, RotationCostModeOverride

        rc = RotationCostConfig(
            enabled=True,
            min_hold_days=7,
            score_gap_threshold=0.05,
            weekly_swap_cap=4,
            per_mode={"swing": RotationCostModeOverride(min_hold_days=0, score_gap_threshold=0.0)},
        )
        s = rc.for_mode("swing")
        assert s.min_hold_days == 0  # 覆蓋
        assert s.score_gap_threshold == 0.0  # 覆蓋
        assert s.enabled is True  # 沿用全域
        assert s.weekly_swap_cap == 4  # 沿用全域
        # momentum 仍走全域
        assert rc.for_mode("momentum").score_gap_threshold == 0.05

    def test_per_mode_loads_from_yaml(self, tmp_path):
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(
            "quant:\n"
            "  rotation_cost:\n"
            "    enabled: true\n"
            "    score_gap_threshold: 0.05\n"
            "    per_mode:\n"
            "      swing:\n"
            "        score_gap_threshold: 0.0\n",
            encoding="utf-8",
        )
        result = load_settings(config_file)
        assert result.quant.rotation_cost.for_mode("swing").score_gap_threshold == 0.0
        assert result.quant.rotation_cost.for_mode("momentum").score_gap_threshold == 0.05


# ---------------------------------------------------------------------------
# A5: settings 拆檔（quant_params.yaml + secrets.yaml 雙檔模式）
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_nested_override(self):
        base = {"a": {"x": 1, "y": 2}, "b": 1}
        overlay = {"a": {"y": 99, "z": 3}, "c": 2}
        merged = _deep_merge(base, overlay)
        assert merged == {"a": {"x": 1, "y": 99, "z": 3}, "b": 1, "c": 2}

    def test_empty_overlay_returns_base(self):
        base = {"a": {"x": 1}}
        assert _deep_merge(base, {}) == base

    def test_scalar_replaces_dict(self):
        # overlay 型別與 base 不同時直接取代（不嘗試合併）
        assert _deep_merge({"a": {"x": 1}}, {"a": "flat"}) == {"a": "flat"}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        overlay = {"a": {"y": 2}}
        _deep_merge(base, overlay)
        assert base == {"a": {"x": 1}}
        assert overlay == {"a": {"y": 2}}


class TestSettingsSplit:
    def _no_legacy(self, monkeypatch, tmp_path):
        """把 CONFIG_PATH 指向不存在的檔，隔離本機 legacy settings.yaml。"""
        monkeypatch.setattr("src.config.CONFIG_PATH", tmp_path / "no_settings.yaml")

    def test_two_file_mode_secrets_override_params(self, monkeypatch, tmp_path):
        self._no_legacy(monkeypatch, tmp_path)
        qp = tmp_path / "quant_params.yaml"
        qp.write_text(
            "finmind:\n  api_url: 'https://example.com'\nlogging:\n  level: DEBUG\n",
            encoding="utf-8",
        )
        sec = tmp_path / "secrets.yaml"
        sec.write_text("finmind:\n  api_token: sekret\n", encoding="utf-8")
        result = load_settings(quant_params_path=qp, secrets_path=sec)
        # 巢狀合併：同區塊內 secrets 補 token、quant_params 的 api_url 保留
        assert result.finmind.api_token == "sekret"
        assert result.finmind.api_url == "https://example.com"
        assert result.logging.level == "DEBUG"

    def test_two_file_mode_both_missing_returns_defaults(self, monkeypatch, tmp_path):
        self._no_legacy(monkeypatch, tmp_path)
        result = load_settings(
            quant_params_path=tmp_path / "nope1.yaml",
            secrets_path=tmp_path / "nope2.yaml",
        )
        assert isinstance(result, Settings)
        assert result.finmind.api_token == ""

    def test_legacy_settings_yaml_takes_precedence(self, monkeypatch, tmp_path):
        legacy = tmp_path / "settings.yaml"
        legacy.write_text("logging:\n  level: WARNING\n", encoding="utf-8")
        monkeypatch.setattr("src.config.CONFIG_PATH", legacy)
        qp = tmp_path / "quant_params.yaml"
        qp.write_text("logging:\n  level: DEBUG\n", encoding="utf-8")
        result = load_settings(quant_params_path=qp, secrets_path=tmp_path / "nope.yaml")
        # legacy 存在 → 完全走舊模式，雙檔被忽略
        assert result.logging.level == "WARNING"

    def test_monitoring_healthchecks_nested_parse(self, monkeypatch, tmp_path):
        """regression：healthchecks_url 必須用巢狀寫法才解析得到。

        原 settings.yaml 曾以 flat dotted key（`monitoring.healthchecks_url: ...`）
        書寫，Pydantic 視為未知頂層 key 靜默忽略 → dead-man 告警靜默失效。
        """
        self._no_legacy(monkeypatch, tmp_path)
        sec = tmp_path / "secrets.yaml"
        sec.write_text(
            "monitoring:\n  healthchecks_url: 'https://hc-ping.com/abc'\n",
            encoding="utf-8",
        )
        result = load_settings(quant_params_path=tmp_path / "nope.yaml", secrets_path=sec)
        assert result.monitoring.healthchecks_url == "https://hc-ping.com/abc"

    def test_flat_dotted_key_is_ignored(self, monkeypatch, tmp_path):
        """flat dotted key 寫法不會被解析（documenting 已知行為，防回歸誤解）。"""
        self._no_legacy(monkeypatch, tmp_path)
        sec = tmp_path / "secrets.yaml"
        sec.write_text(
            "monitoring.healthchecks_url: 'https://hc-ping.com/abc'\n",
            encoding="utf-8",
        )
        result = load_settings(quant_params_path=tmp_path / "nope.yaml", secrets_path=sec)
        assert result.monitoring.healthchecks_url == ""
