import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabjudge.pipeline.clients import JEVClient, LLMClient
from fabjudge.pipeline.config import JEV_KEY_ENV, LLM_KEY_ENV
from fabjudge.pipeline.keys import (
    CredentialConfigurationError, load_jev_key, load_llm_key,
)


class TestRoleSeparatedKeys(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_llm_client_role_rejects_jev_only_environment(self):
        with patch.dict(os.environ, {JEV_KEY_ENV: "jev-test-value"}, clear=True):
            with self.assertRaises(CredentialConfigurationError) as caught:
                LLMClient(self.root, self.root / "llm-cache", max_new_calls=1)
        self.assertIn(LLM_KEY_ENV, str(caught.exception))
        self.assertNotIn("jev-test-value", str(caught.exception))

    def test_jev_client_role_rejects_llm_only_environment(self):
        with patch.dict(os.environ, {LLM_KEY_ENV: "llm-test-value"}, clear=True):
            with self.assertRaises(CredentialConfigurationError) as caught:
                JEVClient(self.root, self.root / "jev-cache", max_new_calls=1)
        self.assertIn(JEV_KEY_ENV, str(caught.exception))
        self.assertNotIn("llm-test-value", str(caught.exception))

    def test_same_key_for_both_roles_is_rejected(self):
        with patch.dict(os.environ, {JEV_KEY_ENV: "same-test-value", LLM_KEY_ENV: "same-test-value"}, clear=True):
            with self.assertRaises(CredentialConfigurationError) as caught:
                load_jev_key(self.root)
        self.assertNotIn("same-test-value", str(caught.exception))

    def test_role_loaders_read_only_their_exact_variable(self):
        values = {JEV_KEY_ENV: "jev-test-value", LLM_KEY_ENV: "llm-test-value"}
        with patch.dict(os.environ, values, clear=True):
            self.assertEqual(load_jev_key(self.root), values[JEV_KEY_ENV])
            self.assertEqual(load_llm_key(self.root), values[LLM_KEY_ENV])

    def test_call_record_contains_role_and_variable_name_but_no_key_value(self):
        with patch.dict(os.environ, {JEV_KEY_ENV: "jev-secret-value", LLM_KEY_ENV: "llm-secret-value"}, clear=True):
            record = JEVClient._attempt_row("request-hash", {
                "status": "success", "cache_hit": False,
                "latency_seconds": 0.1, "cost_usd": 0.001, "error": None,
            })
        serialized = json.dumps(record)
        self.assertEqual(record["key_role"], "jev")
        self.assertEqual(record["key_env_var"], JEV_KEY_ENV)
        self.assertNotIn("jev-secret-value", serialized)
        self.assertNotIn("llm-secret-value", serialized)


if __name__ == "__main__":
    unittest.main()
