"""Offline transport, structured output, and per-sample retry checks."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import requests
from eviground.api import client
from eviground.api.response_schemas import obj, nullable, TEXT
from eviground.grounding.recovery import sample_recovery


class APITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        env = patch.dict(
            os.environ,
            {
                "EVIGROUND_API_KEY": "synthetic-test-key",
                "EVIGROUND_BASE_URL": "https://example.invalid/v1",
                "EVIGROUND_MODEL": "synthetic-model",
                "EVIGROUND_API_PROVIDER": "openai",
                "EVIGROUND_ENABLE_THINKING": "0",
                "EVIGROUND_OPENAI_TOKEN_PARAMETER": "max_completion_tokens",
                "EVIGROUND_OPENAI_TEMPERATURE": "0",
                "EVIGROUND_API_CONCURRENCY": "2",
                "EVIGROUND_API_SLOT_DIR": str(self.root / "slots"),
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.schema = {
            "name": "test_choice",
            "strict": True,
            "schema": obj({"choice": nullable(TEXT)}),
        }
        self.messages = [{"role": "user", "content": "synthetic input"}]
        token = sample_recovery.set({"used": False, "attempts": []})
        self.addCleanup(sample_recovery.reset, token)

    def transport(self, content, finish="stop"):
        data = {
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"total_tokens": 3},
        }
        return SimpleNamespace(
            ok=True,
            status_code=200,
            headers={},
            text=json.dumps(data),
            json=lambda: data,
        )

    def test_structured_completion_uses_compatible_protocol(self):
        with patch(
            "eviground.api.client.requests.post",
            return_value=self.transport('{"choice":"chair"}'),
        ) as post:
            result = client.chat(
                self.messages,
                max_tokens=500,
                json_mode=True,
                response_schema=self.schema,
            )
        self.assertEqual(json.loads(result["_content"]), {"choice": "chair"})
        self.assertEqual(len(result["_request"]["api_attempts"]), 1)
        sent = post.call_args.kwargs
        self.assertEqual(sent["json"]["response_format"]["type"], "json_schema")
        self.assertEqual(sent["json"]["max_completion_tokens"], 500)

    def test_nullable_omission_is_filled_without_inventing_content(self):
        value, changed = client.repair_missing_nullable_fields(
            {}, self.schema["schema"]
        )
        self.assertEqual(value, {"choice": None})
        self.assertEqual(changed, ["choice"])

    def test_invalid_schema_gets_one_feedback_attempt(self):
        values = [self.transport('{"choice":12}'), self.transport('{"choice":"chair"}')]
        with patch("eviground.api.client.requests.post", side_effect=values) as post:
            result = client.chat(
                self.messages, json_mode=True, response_schema=self.schema
            )
        self.assertEqual(post.call_count, 2)
        self.assertEqual(len(result["_request"]["api_attempts"]), 2)

    def test_truncated_response_is_not_retried_as_schema_repair(self):
        with patch(
            "eviground.api.client.requests.post",
            return_value=self.transport("{}", "length"),
        ) as post:
            with self.assertRaises(client.APIError):
                client.chat(self.messages, json_mode=True, response_schema=self.schema)
        self.assertEqual(post.call_count, 1)

    def test_transient_retry_budget_is_shared_across_sample(self):
        with patch(
            "eviground.api.client.requests.post",
            side_effect=[requests.Timeout(), self.transport("{}")],
        ) as post:
            client.chat(self.messages, json_mode=True)
        self.assertEqual(post.call_count, 2)
        self.assertTrue(sample_recovery.get()["used"])
        with patch(
            "eviground.api.client.requests.post", side_effect=requests.Timeout()
        ) as post:
            with self.assertRaises(client.APIError):
                client.chat([{"role": "user", "content": "next step"}], json_mode=True)
        self.assertEqual(post.call_count, 1)

    def test_auth_error_is_not_retried(self):
        failure = SimpleNamespace(
            ok=False, status_code=401, headers={}, text="unauthorized"
        )
        with patch("eviground.api.client.requests.post", return_value=failure) as post:
            with self.assertRaises(client.APIError):
                client.chat(self.messages, json_mode=True)
        self.assertEqual(post.call_count, 1)

    def test_completed_request_replay_is_confined_to_explicit_same_sample_recovery(
        self,
    ):
        budget = sample_recovery.get()
        budget.update(
            api_checkpoint=str(self.root / "recovery.json"),
            checkpoint_identity="synthetic-identity",
            completed_requests={},
            resuming=False,
        )
        with patch(
            "eviground.api.client.requests.post", return_value=self.transport("{}")
        ) as post:
            client.chat(self.messages, json_mode=True)
            budget["resuming"] = True
            replay = client.chat(self.messages, json_mode=True)
        self.assertEqual(post.call_count, 1)
        self.assertTrue(replay["_request"]["cache_hit"])
        self.assertIsNone(replay["usage"])
        sample_recovery.set({"used": False, "attempts": []})
        with patch(
            "eviground.api.client.requests.post", return_value=self.transport("{}")
        ) as post:
            client.chat(self.messages, json_mode=True)
        self.assertEqual(post.call_count, 1)

    def test_model_key_and_inline_images_are_redacted(self):
        redacted = client._redact(
            {
                "api_key": "synthetic-test-key",
                "text": "Bearer synthetic-test-key",
                "image_url": {"url": "data:image/png;base64,c2VjcmV0"},
            }
        )
        self.assertNotIn("synthetic-test-key", json.dumps(redacted))
        self.assertNotIn("c2VjcmV0", json.dumps(redacted))

    def test_thinking_uses_configured_token_floor(self):
        with patch.dict(
            os.environ,
            {
                "EVIGROUND_ENABLE_THINKING": "1",
                "EVIGROUND_THINKING_TOKEN_FLOOR": "8192",
            },
        ):
            self.assertEqual(
                client.request_options(500, None, True)["max_completion_tokens"], 8192
            )

    def test_obsolete_provider_is_rejected(self):
        with (
            patch.dict(os.environ, {"EVIGROUND_API_PROVIDER": "qwen"}),
            self.assertRaises(client.APIError),
        ):
            client.request_options(500, None, True)

    def test_invalid_limits_and_temperature_are_rejected(self):
        for value in ("0", "-1", "invalid"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"EVIGROUND_API_CONCURRENCY": value}),
                self.assertRaises(client.APIError),
            ):
                client.api_concurrency()
        with self.assertRaises(client.APIError):
            client.request_options(500, float("nan"), True)
