import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from llm_agent import cli
from llm_agent.service import ConversationService
from tests.helpers import completion


class ModelCLIManagementTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.output = io.StringIO()
        self.enterContext(patch.object(cli, "console", Console(file=self.output, width=200, color_system=None)))
        self.enterContext(patch.object(cli, "load_env"))

    def run_cli(self, *args):
        self.output.seek(0)
        self.output.truncate()
        with patch("sys.argv", ["agent", "--data-dir", str(self.directory), *args]), patch("sys.stdout", self.output):
            cli.main()
        return self.output.getvalue()

    def test_catalog_starts_empty_despite_env_key(self):
        with patch.dict(os.environ, {"API_KEY": "cloud-secret"}):
            self.assertEqual(json.loads(self.run_cli("--models-list")), [])

    def test_add_local_select_and_send_without_env_token(self):
        added = json.loads(self.run_cli("--model-add", "gemma", "--base-url", "http://localhost:1234/v1"))
        self.assertEqual(added[0]["name"], "gemma")
        self.assertFalse(added[0]["has_token"])
        self.run_cli("--new-conversation", "--model", "gemma")
        service = ConversationService(self.directory)
        conversation = service.list_conversations()[0]
        self.assertEqual(conversation.selected_model_id, added[0]["id"])
        with patch("llm_agent.agent.OpenAI") as factory, patch("llm_agent.llm_client.DefaultHttpxClient"):
            factory.return_value.chat.completions.create.return_value = completion()
            result = json.loads(self.run_cli("--conversation", conversation.id, "--batch", "--user", "Hello"))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(factory.call_args.kwargs["api_key"], "local-no-auth")

    def test_explicit_token_env_is_saved_but_not_printed(self):
        with patch.dict(os.environ, {"CATALOG_TEST_TOKEN": "new-secret"}):
            output = self.run_cli("--model-add", "cloud", "--base-url", "https://example.com/v1",
                                  "--model-token-env", "CATALOG_TEST_TOKEN", "--model-name", "Cloud")
        self.assertNotIn("new-secret", output)
        record = ConversationService(self.directory).models.list()[0]
        self.assertEqual(record.token, "new-secret")
        self.assertEqual(record.name, "Cloud")

    def test_discovery_does_not_add_models_and_import_is_idempotent(self):
        with patch.object(cli, "discover_models", return_value=["gemma", "other"]) as discover:
            output = self.run_cli("--models-discover", "http://localhost:1234/v1")
            self.assertEqual(json.loads(output), ["gemma", "other"])
            self.assertEqual(ConversationService(self.directory).models.list(), [])
            first = json.loads(self.run_cli("--models-import", "http://localhost:1234/v1"))
            second = json.loads(self.run_cli("--models-import", "http://localhost:1234/v1"))
            self.assertEqual(first, second)
            discover.assert_called_with("http://localhost:1234/v1", "")

    def test_conflicting_actions_are_rejected_before_mutations(self):
        for args in (("--models-list", "--new-conversation"), ("--models-list", "--profile-list"),
                     ("--model-add", "gemma"), ("--base-url", "http://localhost/v1")):
            with self.subTest(args=args), self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
                self.run_cli(*args)
        self.assertEqual(ConversationService(self.directory).models.list(), [])


if __name__ == "__main__":
    unittest.main()
