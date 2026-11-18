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
from eviground.api.response_schemas import obj, nullable, TEXT, REASON
from eviground.grounding.recovery import sample_recovery


class LocalModelDefaultsTests(unittest.TestCase):
    def test_guidance_react_contract_is_not_duplicated_in_prompt(self):
        from copy import deepcopy

        messages = [{"role": "system", "content": "Read all evidence and update the DAG."},
                    {"role": "user", "content": [{"type": "text", "text": "observations"},
                                                {"type": "image_url", "image_url": {"url": "synthetic"}}]}]
        original = deepcopy(messages)
        schema = dict(name="react_action", strict=True, schema=obj({"choice": TEXT}))
        with patch.dict(os.environ, {"EVIGROUND_PROMPT_PROFILE": "compact",
                                     "EVIGROUND_GUIDED_DECODING_BACKEND": "guidance"}):
            result = client._schema_messages(messages, schema, "choose an existing ID")
            options = client.request_options(4000, None, True, schema)
            parser = client._schema_messages(messages, {**schema, "name": "query_parser"})
        self.assertNotIn("Output JSON Schema:", result[0]["content"])
        self.assertIn("choose an existing ID", result[0]["content"])
        self.assertTrue(result[0]["content"].startswith(messages[0]["content"]))
        self.assertEqual(result[1:], messages[1:])
        self.assertEqual(messages, original)
        self.assertEqual(options["response_format"]["json_schema"], schema)
        self.assertEqual(options["guided_decoding_backend"], "guidance")
        self.assertIn("Output JSON Schema:", parser[0]["content"])

    def test_guidance_backend_is_explicit_and_audited(self):
        with patch.dict(os.environ, {"EVIGROUND_GUIDED_DECODING_BACKEND": "guidance"}):
            options = client.request_options(100, None, True)
            self.assertEqual(options["guided_decoding_backend"], "guidance")
            self.assertEqual(client.api_config_summary()["request_options"]["guided_decoding_backend"], "guidance")
            self.assertNotIn("guided_decoding_backend", client.request_options(100, None, False))
        with patch.dict(os.environ, {"EVIGROUND_GUIDED_DECODING_BACKEND": "xgrammar"}):
            with self.assertRaisesRegex(client.APIError, "empty or guidance"):
                client.request_options(100, None, True)

    def test_compact_parser_bounds_prevent_category_and_clause_enumeration(self):
        from eviground.api.response_schemas import parser_schema

        compact = parser_schema(compact=True)["properties"]
        self.assertEqual(compact["Target"]["maxItems"], 1)
        for name in ["Anchor", "Relations", "Attributes", "Unhandled"]:
            self.assertEqual(compact[name]["maxItems"], 8)
        self.assertNotIn("maxItems", parser_schema()["properties"]["Target"])
        standard_relation = parser_schema()["properties"]["Relations"]["items"]["properties"]
        self.assertNotIn("maxItems", standard_relation["anchor_categories"])

    def test_compact_contract_preserves_final_multimodal_user_turn(self):
        from copy import deepcopy

        messages = [{"role": "system", "content": "Locate the described object."},
                    {"role": "user", "content": [{"type": "text", "text": "original task"},
                                                {"type": "image_url", "image_url": {"url": "synthetic"}}]}]
        original = deepcopy(messages)
        schema = {"name": "choice", "strict": True, "schema": obj({"choice": TEXT})}
        with patch.dict(os.environ, {"EVIGROUND_PROMPT_PROFILE": "compact"}):
            result = client._schema_messages(messages, schema, "choice must be a string")
            options = client.request_options(100, None, True, schema)
        self.assertEqual(messages, original)
        self.assertEqual(result[-1], messages[-1])
        self.assertIn("choice must be a string", result[0]["content"])
        textual_schema = result[0]["content"].split("Output JSON Schema: ", 1)[1].split(
            " Previous validation error:", 1)[0]
        self.assertEqual(json.loads(textual_schema), schema["schema"])
        self.assertEqual(options["response_format"]["json_schema"], schema)

    def test_factored_prompt_schema_expands_exactly_and_keeps_decoder(self):
        from copy import deepcopy
        from jsonschema import Draft202012Validator

        repeated = obj({"ids": {"type": "array", "items": {"enum": list(range(40))}},
                        "conclusion": REASON})
        original = obj({"a": repeated, "b": deepcopy(repeated),
                        "literal": {"const": {"type": "object", "text": "literal" * 40}}})
        original["$defs"] = {"text_part_0": {"type": "null"}}
        schema = dict(name="react_action", strict=True, schema=original)
        before = deepcopy(schema)
        messages = [{"role": "system", "content": "Read all evidence."}]
        with patch.dict(os.environ, {"EVIGROUND_PROMPT_PROFILE": "compact"}):
            result = client._schema_messages(messages, schema)
            options = client.request_options(4000, None, True, schema)
        factored = json.loads(result[0]["content"].split("Output JSON Schema: ", 1)[1])
        new_names = set(factored["$defs"]) - set(original["$defs"])
        self.assertTrue(new_names)

        def expand(value):
            if isinstance(value, list):
                return [expand(item) for item in value]
            if not isinstance(value, dict):
                return value
            if set(value) == {"$ref"} and value["$ref"].split("/")[-1] in new_names:
                return expand(factored["$defs"][value["$ref"].split("/")[-1]])
            return {key: expand(item) for key, item in value.items()}

        restored = expand({**factored, "$defs": {k: v for k, v in factored["$defs"].items()
                                                 if k not in new_names}})
        self.assertEqual(restored, client._wire_schema(original))
        Draft202012Validator.check_schema(factored)
        self.assertLess(len(json.dumps(factored)), len(json.dumps(original)))
        self.assertEqual(schema, before)
        self.assertEqual(options["response_format"]["json_schema"], client._wire_schema(schema))

    def test_prompt_profile_is_audited_and_invalid_values_fail(self):
        from eviground.api.prompts import prompt_profile

        with patch.dict(os.environ, {"EVIGROUND_PROMPT_PROFILE": "compact"}):
            self.assertEqual(client.api_config_summary()["prompt_profile"], "compact")
        with patch.dict(os.environ, {"EVIGROUND_PROMPT_PROFILE": "unknown"}):
            with self.assertRaises(ValueError):
                prompt_profile()

    def test_reason_contract_accepts_sentences_and_rejects_whitespace(self):
        import re
        import jsonschema

        for value in ["The chair is beside the table.", "\nEvidence supports chair 2.\n", "有可见证据"]:
            jsonschema.validate(value, REASON)
            self.assertIsNotNone(re.fullmatch(REASON["pattern"], value))
        for value in ["", " ", "\n\t"]:
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(value, REASON)

    def test_local_defaults_send_vllm_parameters(self):
        from eviground.cli.entrypoint import configure

        with patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1"}, clear=True):
            configure()
            options = client.request_options(1600, None, True)
            self.assertEqual(os.environ["EVIGROUND_MODEL"], "qwen2-vl-72b")
            self.assertEqual(os.environ["EVIGROUND_BASE_URL"], "http://127.0.0.1:18000/v1")
            self.assertEqual(options["max_tokens"], 1600)
            self.assertFalse(options["enable_thinking"])
            self.assertNotIn("max_completion_tokens", options)

    def test_explicit_model_and_endpoint_override_defaults(self):
        from eviground.cli.entrypoint import configure

        with patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1",
                                   "EVIGROUND_MODEL": "qwen2-vl-2b",
                                   "EVIGROUND_BASE_URL": "http://127.0.0.1:28000/v1"}, clear=True):
            configure()
            self.assertEqual(os.environ["EVIGROUND_MODEL"], "qwen2-vl-2b")
            self.assertEqual(os.environ["EVIGROUND_BASE_URL"], "http://127.0.0.1:28000/v1")


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

    def test_local_attempt_journal_does_not_change_model_payload(self):
        from eviground.api.usage_log import read_attempts

        payloads = []
        for enabled in (False, True):
            budget = dict(used=False, attempts=[])
            if enabled:
                budget.update(api_checkpoint=str(self.root / 'api_recovery.json'),
                              call_stage='parse_query', execution_id='test-execution')
            token = sample_recovery.set(budget)
            try:
                with patch('eviground.api.client.requests.post',
                           return_value=self.transport('{"choice":"chair"}')) as post:
                    client.chat(self.messages, max_tokens=500, json_mode=True, response_schema=self.schema)
                payloads.append(post.call_args.kwargs['json'])
            finally:
                sample_recovery.reset(token)
        self.assertEqual(payloads[0], payloads[1])
        attempts = read_attempts(self.root / 'http_attempts.jsonl')
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['stage'], 'parse_query')
        self.assertEqual(attempts[0]['usage'], dict(total_tokens=3))
        self.assertNotIn('test-execution', json.dumps(payloads))

    def test_local_read_timeout_is_configurable_and_explicit_timeout_wins(self):
        with (patch.dict(os.environ, {"EVIGROUND_API_READ_TIMEOUT": "900"}),
              patch("eviground.api.client.requests.post",
                    return_value=self.transport('{"choice":"chair"}')) as post):
            client.chat(self.messages, json_mode=True, response_schema=self.schema)
            self.assertEqual(post.call_args.kwargs["timeout"], (15, 900))
            self.assertEqual(client.api_config_summary()["http_read_timeout_s"], 900)
            client.chat(self.messages, timeout=(3, 20), json_mode=True, response_schema=self.schema)
            self.assertEqual(post.call_args.kwargs["timeout"], (3, 20))
        with patch.dict(os.environ, {"EVIGROUND_API_READ_TIMEOUT": "0"}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                client.chat(self.messages)

    def test_nonblank_reason_is_validated_locally_without_wildcard_grammar(self):
        from copy import deepcopy

        schema = {"name": "reason", "strict": True, "schema": obj({"reason": REASON})}
        original = deepcopy(schema)
        with patch("eviground.api.client.requests.post",
                   return_value=self.transport(json.dumps({"reason": 'The "lower" box.'}))) as post:
            result = client._chat_once(self.messages, json_mode=True, response_schema=schema)
        self.assertEqual(json.loads(result["_content"])["reason"], 'The "lower" box.')
        wire = post.call_args.kwargs["json"]["response_format"]["json_schema"]
        self.assertEqual(wire["schema"]["properties"]["reason"], {"type": "string", "minLength": 1})
        self.assertEqual(schema, original)
        with patch("eviground.api.client.requests.post",
                   return_value=self.transport('{"reason":"   "}')):
            with self.assertRaises(client.APIError) as caught:
                client._chat_once(self.messages, json_mode=True, response_schema=schema)
        self.assertEqual(caught.exception.diagnostic["stage"], "response_schema_validate")

    def test_nullable_omission_is_filled_without_inventing_content(self):
        value, changed = client.repair_missing_nullable_fields(
            {}, self.schema["schema"]
        )
        self.assertEqual(value, {"choice": None})
        self.assertEqual(changed, ["choice"])

    def test_tuple_array_boolean_items_reaches_strict_validation(self):
        schema = dict(name="initial_tasks", strict=True, schema=obj({"nodes": {
            "type": "array", "prefixItems": [obj({"id": {"enum": ["n0"]}})],
            "items": False, "minItems": 1, "maxItems": 1}}))
        with patch("eviground.api.client.requests.post",
                   return_value=self.transport('{"nodes":[{"id":"n0"}]}')):
            result = client._chat_once(self.messages, json_mode=True, response_schema=schema)
        self.assertEqual(json.loads(result["_content"]), {"nodes": [{"id": "n0"}]})
        for invalid in ('{"nodes":[{"id":"n1"}]}', '{"nodes":[{"id":"n0"},{"id":"n0"}]}'):
            with patch("eviground.api.client.requests.post", return_value=self.transport(invalid)):
                with self.assertRaises(client.APIError) as caught:
                    client._chat_once(self.messages, json_mode=True, response_schema=schema)
            self.assertEqual(caught.exception.diagnostic["stage"], "response_schema_validate")

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

    def test_http_500_recovers_once_without_restarting_the_sample(self):
        failure = SimpleNamespace(ok=False, status_code=500, headers={}, text="internal error")
        with patch(
            "eviground.api.client.requests.post",
            side_effect=[failure, self.transport('{"choice":"chair"}')],
        ) as post:
            result = client.chat(self.messages, json_mode=True)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(json.loads(result["_content"]), {"choice": "chair"})
        self.assertTrue(sample_recovery.get()["used"])
        self.assertEqual(len(result["_request"]["api_attempts"]), 2)

    def test_persistent_http_500_stops_after_one_recovery_attempt(self):
        failure = SimpleNamespace(ok=False, status_code=500, headers={}, text="internal error")
        with patch("eviground.api.client.requests.post", return_value=failure) as post:
            with self.assertRaises(client.APIError):
                client.chat(self.messages, json_mode=True)
        self.assertEqual(post.call_count, 2)

    def test_provider_aborted_json_generation_recovers_the_same_request_once(self):
        body = json.dumps({"error": {"code": "invalid_parameter_error", "message":
            "Model output became abnormal while generating a JSON response for response_format. The generation was aborted because the partial output may be incomplete or invalid JSON."}})
        failure = SimpleNamespace(ok=False, status_code=400, headers={}, text=body)
        with patch("eviground.api.client.requests.post",
                   side_effect=[failure, self.transport('{"choice":"chair"}')]) as post:
            result = client.chat(self.messages, json_mode=True, response_schema=self.schema)
        self.assertEqual(json.loads(result["_content"]), {"choice": "chair"})
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs["json"], post.call_args_list[1].kwargs["json"])
        self.assertTrue(sample_recovery.get()["used"])

    def test_provider_aborted_json_generation_does_not_retry_indefinitely(self):
        body = json.dumps({"error": {"code": "invalid_parameter_error", "message":
            "Model output became abnormal while generating a JSON response for response_format."}})
        failure = SimpleNamespace(ok=False, status_code=400, headers={}, text=body)
        with patch("eviground.api.client.requests.post", return_value=failure) as post:
            with self.assertRaises(client.APIError):
                client.chat(self.messages, json_mode=True, response_schema=self.schema)
        self.assertEqual(post.call_count, 2)

    def test_other_http_400_parameter_errors_are_not_retried(self):
        failure = SimpleNamespace(ok=False, status_code=400, headers={},
            text=json.dumps({"error": {"code": "invalid_parameter_error", "message": "Input token limit exceeded"}}))
        with patch("eviground.api.client.requests.post", return_value=failure) as post:
            with self.assertRaises(client.APIError):
                client.chat(self.messages, json_mode=True, response_schema=self.schema)
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
