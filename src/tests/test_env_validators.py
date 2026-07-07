"""Unit tests for EnvHelper validators and utility logic.

Tests the pure Python logic in src/env.py without requiring a live Azure Key Vault:
- transform_REST_API_KEYS: JSON string → list conversion (Terraform single-quote fix)
- validate_ENVIRONMENT: only "DEV" or "PRODUCTION" accepted
- append_variable: env-var lookup priority over key vault; hyphens in vault key names
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import ResourceNotFoundError

from src.env import EnvHelper


# ── transform_REST_API_KEYS ───────────────────────────────────────────────────

class TestTransformRESTAPIKeys:
    """The @field_validator converts JSON strings (possibly single-quoted) to lists."""

    def _call(self, value):
        return EnvHelper.transform_REST_API_KEYS(value)

    def test_list_passed_through_unchanged(self):
        assert self._call(["key1", "key2"]) == ["key1", "key2"]

    def test_empty_list_allowed(self):
        assert self._call([]) == []

    def test_json_string_double_quoted(self):
        assert self._call('["k1", "k2"]') == ["k1", "k2"]

    def test_json_string_empty_array(self):
        assert self._call("[]") == []

    def test_single_quoted_terraform_style(self):
        # Terraform outputs single-quoted JSON that is not valid JSON — we fix it.
        assert self._call("['key_a', 'key_b']") == ["key_a", "key_b"]

    def test_single_item_json_list(self):
        assert self._call('["only"]') == ["only"]

    def test_non_list_non_string_raises_value_error(self):
        with pytest.raises((ValueError, Exception)):
            self._call(42)

    def test_invalid_json_string_raises(self):
        with pytest.raises(Exception):
            self._call("{not-valid-json")


# ── validate_ENVIRONMENT ──────────────────────────────────────────────────────

class TestValidateEnvironment:
    """Only 'DEV' and 'PRODUCTION' are accepted."""

    def _call(self, value):
        return EnvHelper.validate_ENVIRONMENT(value)

    def test_dev_accepted(self):
        assert self._call("DEV") == "DEV"

    def test_production_accepted(self):
        assert self._call("PRODUCTION") == "PRODUCTION"

    def test_lowercase_dev_rejected(self):
        with pytest.raises(ValueError):
            self._call("dev")

    def test_lowercase_production_rejected(self):
        with pytest.raises(ValueError):
            self._call("production")

    def test_staging_rejected(self):
        with pytest.raises(ValueError):
            self._call("STAGING")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            self._call("")

    def test_arbitrary_string_rejected(self):
        with pytest.raises(ValueError):
            self._call("TEST")


# ── append_variable ───────────────────────────────────────────────────────────

class TestAppendVariable:
    """Static method: reads from env-var or falls back to Azure Key Vault."""

    # Helper: a unique key name that is almost certainly not in the real environment
    _KEY = "KIC_TEST_UNIQUE_VAR_7F9A"

    def _stripped_env(self):
        """Environment dict without our test key."""
        return {k: v for k, v in os.environ.items() if k != self._KEY}

    def test_env_var_takes_priority_over_key_vault(self):
        kwargs = {}
        mock_client = MagicMock()
        with patch.dict(os.environ, {self._KEY: "from_env"}):
            result = EnvHelper.append_variable(kwargs, self._KEY, secret_client=mock_client)
        assert result[self._KEY] == "from_env"
        mock_client.get_secret.assert_not_called()

    def test_falls_back_to_key_vault_when_env_var_absent(self):
        kwargs = {}
        mock_client = MagicMock()
        mock_client.get_secret.return_value.value = "from_vault"
        with patch.dict(os.environ, self._stripped_env(), clear=True):
            result = EnvHelper.append_variable(kwargs, self._KEY, secret_client=mock_client)
        assert result[self._KEY] == "from_vault"

    def test_key_not_in_vault_leaves_kwargs_unchanged(self):
        kwargs = {}
        mock_client = MagicMock()
        mock_client.get_secret.side_effect = ResourceNotFoundError("not found")
        with patch.dict(os.environ, self._stripped_env(), clear=True):
            result = EnvHelper.append_variable(kwargs, self._KEY, secret_client=mock_client)
        assert self._KEY not in result
        assert result == {}

    def test_class_variable_overrides_kwargs_key_name(self):
        """class_variable lets callers map an env key to a different model field name."""
        kwargs = {}
        mock_client = MagicMock()
        with patch.dict(os.environ, {self._KEY: "value"}):
            result = EnvHelper.append_variable(
                kwargs, self._KEY, secret_client=mock_client, class_variable="DIFFERENT_KEY"
            )
        assert result["DIFFERENT_KEY"] == "value"
        assert self._KEY not in result

    def test_vault_lookup_uses_hyphens_not_underscores(self):
        """Azure Key Vault names cannot contain underscores — they must be replaced."""
        kwargs = {}
        mock_client = MagicMock()
        mock_client.get_secret.return_value.value = "val"
        with patch.dict(os.environ, self._stripped_env(), clear=True):
            EnvHelper.append_variable(kwargs, self._KEY, secret_client=mock_client)
        called_key = mock_client.get_secret.call_args[0][0]
        assert "_" not in called_key, f"Vault key still contains underscores: {called_key!r}"
        assert "-" in called_key

    def test_returns_kwargs_dict(self):
        """append_variable returns the mutated kwargs dict."""
        kwargs = {}
        with patch.dict(os.environ, {self._KEY: "x"}):
            result = EnvHelper.append_variable(kwargs, self._KEY, secret_client=MagicMock())
        assert result is kwargs
