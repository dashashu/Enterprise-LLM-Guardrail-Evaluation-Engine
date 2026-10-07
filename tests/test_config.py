"""Configuration requirements differ between the HTTP API and eval CLI."""

import pytest

from app import config as config_module
from app.config import Settings

DEFAULT_DOTENV_PATH = config_module._DOTENV_PATH


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Keep tests independent of a developer's real .env and shell settings."""
    dotenv_path = tmp_path / ".env"
    monkeypatch.setattr(config_module, "_DOTENV_PATH", dotenv_path)
    for name in ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "APP_API_KEYS", "REDIS_URL", "RATE_LIMIT_PER_MINUTE",
                 "LLM_TIMEOUT_SECONDS", "REQUEST_DEADLINE_SECONDS", "LLM_MAX_RETRIES",
                 "CACHE_TTL_SECONDS", "MODERATION_MODE"):
        monkeypatch.delenv(name, raising=False)
    return dotenv_path


def test_dotenv_path_is_anchored_at_repository_root():
    assert DEFAULT_DOTENV_PATH == config_module.Path(config_module.__file__).resolve().parent.parent / ".env"


def test_settings_load_values_from_dotenv_when_shell_is_unset(isolated_environment, monkeypatch):
    isolated_environment.write_text(
        '# Example configuration\n'
        'LLM_PROVIDER=openai\n'
        'LLM_MODEL="example-model"\n'
        'OPENAI_API_KEY="fixture-key"\n'
        'APP_API_KEYS="client-one, client-two"\n'
        'RATE_LIMIT_PER_MINUTE=42\n',
        encoding="utf-8",
    )
    outside = isolated_environment.parent / "other-directory"
    outside.mkdir()
    monkeypatch.chdir(outside)

    settings = Settings.from_env()

    assert settings.provider == "openai"
    assert settings.model == "example-model"
    assert settings.api_key == "fixture-key"
    assert settings.app_api_keys == ("client-one", "client-two")
    assert settings.rate_limit_per_minute == 42
    assert "OPENAI_API_KEY" not in config_module.os.environ


def test_exported_variables_override_dotenv_values(isolated_environment, monkeypatch):
    isolated_environment.write_text(
        "OPENAI_API_KEY=file-key\nLLM_MODEL=file-model\nAPP_API_KEYS=file-client\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")
    monkeypatch.setenv("LLM_MODEL", "shell-model")
    monkeypatch.setenv("APP_API_KEYS", "shell-client")

    settings = Settings.from_env()

    assert settings.api_key == "shell-key"
    assert settings.model == "shell-model"
    assert settings.app_api_keys == ("shell-client",)


def test_explicitly_empty_export_does_not_fall_back_to_dotenv(isolated_environment, monkeypatch):
    isolated_environment.write_text(
        "OPENAI_API_KEY=file-key\nLLM_MODEL=file-model\n", encoding="utf-8"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings.from_env(require_app_api_keys=False)


def test_explicitly_empty_dotenv_provider_is_invalid(isolated_environment):
    isolated_environment.write_text("LLM_PROVIDER=\n", encoding="utf-8")

    with pytest.raises(ValueError, match="LLM_PROVIDER must be openai or anthropic"):
        Settings.from_env(require_app_api_keys=False)


def test_display_name_is_rejected_as_model_id(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv("LLM_MODEL", "GPT-6 Astra")

    with pytest.raises(ValueError, match="API model ID without spaces"):
        Settings.from_env(require_app_api_keys=False)


@pytest.mark.parametrize("provider,key_name", [("openai", "OPENAI_API_KEY"),
                                                 ("anthropic", "ANTHROPIC_API_KEY")])
def test_evaluation_requires_provider_credentials_but_not_app_api_keys(monkeypatch, provider, key_name):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(key_name, "provider-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("MODERATION_MODE", "local")
    monkeypatch.delenv("APP_API_KEYS", raising=False)

    evaluation = Settings.from_env(require_app_api_keys=False)
    assert evaluation.provider == provider
    assert evaluation.api_key == "provider-key"
    assert evaluation.model == "test-model"
    assert evaluation.app_api_keys == ()

    with pytest.raises(ValueError, match="APP_API_KEYS"):
        Settings.from_env()


@pytest.mark.parametrize("provider,key_name", [("openai", "OPENAI_API_KEY"),
                                                 ("anthropic", "ANTHROPIC_API_KEY")])
def test_evaluation_reports_only_missing_provider_settings(monkeypatch, provider, key_name):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("MODERATION_MODE", "local")
    monkeypatch.delenv(key_name, raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("APP_API_KEYS", raising=False)

    with pytest.raises(ValueError) as exc:
        Settings.from_env(require_app_api_keys=False)

    assert key_name in str(exc.value)
    assert "LLM_MODEL" in str(exc.value)
    assert "APP_API_KEYS" not in str(exc.value)
