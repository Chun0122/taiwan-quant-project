"""測試 src/config.py — 設定載入。"""

from src.config import Settings, load_settings


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
