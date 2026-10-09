"""Настройки и диагностика RAG в CLI без индекса, модели и сети."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from openai.types.chat import ChatCompletion

from llm_agent.agent import RequestResult

from llm_agent.cli import context_settings, parse_args, print_rag_sources, run, send_message
from llm_agent.models import ContextSettings, RAGSettings, RequestOptions, Turn
from llm_agent.service import ConversationService
from tests.helpers import create_selected_conversation


class RAGCLITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.service = ConversationService(self.directory, token=None)
        self.conversation = create_selected_conversation(self.service)
        self.enterContext(patch("llm_agent.agent.OpenAI", side_effect=AssertionError("Сеть запрещена")))
        self.enterContext(patch("llm_agent.cli.load_env"))
        self.enterContext(patch.dict(os.environ, {"API_KEY": "test-only"}))

    def arguments(self, *args):
        return ["main.py", "--conversation", str(self.service.store.path(self.conversation.id)), *args]

    def cli(self, *args):
        output = io.StringIO()
        with patch("sys.argv", self.arguments(*args)), patch("llm_agent.cli.console", Console(file=output, width=240)):
            run()
        return output.getvalue()

    def test_rag_settings_are_saved_for_same_shared_conversation_without_api(self):
        self.cli("--rag", "--rag-rewrite", "--rag-filter", "--rag-top-k-before", "12",
                 "--rag-top-k-after", "3", "--rag-similarity-threshold", "0.77")
        expected = ContextSettings(rag_enabled=True, rag_rewrite_enabled=True, rag_filter_enabled=True,
                                   rag_top_k_before=12, rag_top_k_after=3, rag_similarity_threshold=0.77)
        self.assertEqual(self.service.get(self.conversation.id).settings, expected)
        self.cli("--no-rag")
        self.assertEqual(self.service.get(self.conversation.id).settings, replace(expected, rag_enabled=False))
        self.cli("--no-rag-rewrite", "--no-rag-filter")
        self.assertEqual(self.service.get(self.conversation.id).settings,
                         replace(expected, rag_enabled=False, rag_rewrite_enabled=False, rag_filter_enabled=False))
        self.assertEqual(len(self.service.list_conversations()), 1)
        self.assertEqual(self.service.get(self.conversation.id).turns, [])

    def test_omitted_flags_preserve_all_rag_settings(self):
        current = ContextSettings(rag_enabled=True, rag_rewrite_enabled=True, rag_filter_enabled=True,
                                  rag_top_k_before=40, rag_top_k_after=8, rag_similarity_threshold=0.55)
        with patch("sys.argv", self.arguments("--user", "Продолжим")):
            args = parse_args()
        self.assertEqual(context_settings(args, current), current)

    def test_invalid_values_never_change_existing_conversation(self):
        before = self.service.store.path(self.conversation.id).read_bytes()
        for arguments in (
            ("--rag-top-k-before", "2", "--rag-top-k-after", "3"),
            ("--rag-top-k-after", "1.5"), ("--rag-top-k-before", "0"),
            ("--rag-similarity-threshold", "nan"), ("--rag-similarity-threshold", "inf"),
            ("--rag-similarity-threshold", "1.1"), ("--rag-similarity-threshold", "-1.1"),
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.cli(*arguments)
                self.assertEqual(self.service.store.path(self.conversation.id).read_bytes(), before)

    def test_rag_flags_require_conversation_and_cannot_modify_view_commands(self):
        for arguments in (
            ["--rag-filter"], ["--rag-rewrite"], ["--rag-top-k-after", "2"],
            ["--list-conversations", "--rag-filter"],
            self.arguments("--invariants-show", "--rag-filter")[1:],
            self.arguments("--show-conversation", "--no-rag-rewrite")[1:],
        ):
            with self.subTest(arguments=arguments), patch("sys.argv", ["main.py", *arguments]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_mode_uses_saved_turn_snapshot_and_batch_preserves_it(self):
        result = self.service.get(self.conversation.id)
        result.turns = [Turn(user="Вопрос", answer="Ответ", status="completed", rag_enabled=True,
                             rag_settings=RAGSettings(rewrite_enabled=True, filter_enabled=True))]
        output = io.StringIO()
        with patch("sys.argv", self.arguments("--user", "Вопрос")):
            args = parse_args()
        with patch("llm_agent.cli.ConversationService") as factory, patch("llm_agent.cli.console", Console(file=output)):
            factory.return_value.get_model.return_value = self.service.get_model(self.conversation.id)
            factory.return_value.send.return_value = result
            send_message(args, self.conversation, RequestOptions())
        self.assertIn("Режим: RAG + rewrite + фильтр", output.getvalue())
        self.assertFalse(self.conversation.settings.rag_enabled)
        args.batch = True
        with patch("llm_agent.cli.ConversationService") as factory, contextlib.redirect_stdout(io.StringIO()) as stdout:
            factory.return_value.get_model.return_value = self.service.get_model(self.conversation.id)
            factory.return_value.send.return_value = result
            send_message(args, self.conversation, RequestOptions())
        data = json.loads(stdout.getvalue())
        self.assertEqual(data["rag_settings"], asdict(result.turns[0].rag_settings))
        self.assertTrue(data["rag_enabled"])

    def test_rag_prints_verified_result_and_usage_instead_of_raw_response(self):
        result = self.service.get(self.conversation.id)
        result.turns = [Turn(user="Вопрос", answer="Проверенный ответ", status="completed", rag_enabled=True)]
        output = io.StringIO()
        with patch("sys.argv", self.arguments("--user", "Вопрос")):
            args = parse_args()

        def send(*args, **kwargs):
            response = ChatCompletion(
                id="test", object="chat.completion", created=0, model="test",
                choices=[{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "RAW_UNVERIFIED_JSON",
                }}],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
            kwargs["on_response"](RequestResult(response, 0.1, answer="Проверенный ответ"))
            return result

        with patch("llm_agent.cli.ConversationService") as factory, patch("llm_agent.cli.console", Console(file=output)):
            factory.return_value.get_model.return_value = self.service.get_model(self.conversation.id)
            factory.return_value.send.side_effect = send
            send_message(args, self.conversation, RequestOptions())
        self.assertIn("Проверенный ответ", output.getvalue())
        self.assertIn("Токены за запрос", output.getvalue())
        self.assertNotIn("RAW_UNVERIFIED_JSON", output.getvalue())

    def test_empty_sources_explain_filtering_and_fallback_with_literal_input(self):
        settings = RAGSettings(filter_enabled=True, rewrite_enabled=True, similarity_threshold=0.9)
        context = {
            "index": "test-index", "chunks": [], "settings": asdict(settings),
            "original_query": "[bold]Вопрос[/bold]", "search_query": "[bold]Вопрос[/bold]",
            "rewrite": {"status": "fallback", "reason": "[red]bad[/red]", "elapsed_seconds": 0.2, "usage": None},
            "candidates": [{"source": "[red]file.md[/red]", "section": "", "text": "Текст", "score": 0.8,
                            "chunk_id": "c1", "original_rank": 1, "selection_reason": "below_threshold"}],
            "counts": {"found": 1, "passed": 0, "selected": 0}, "retrieval_seconds": 0.3,
            "embedding": {"model": "test-model", "revision": "abc"}, "corpus_hash": "corpus-hash",
        }
        output = io.StringIO()
        with patch("llm_agent.cli.console", Console(file=output, width=240)):
            print_rag_sources(Turn(user="Вопрос", rag_enabled=True, rag_context=context, rag_settings=settings))
        rendered = output.getvalue()
        for expected in ("релевантных фрагментов не осталось", "[bold]Вопрос[/bold]", "[red]bad[/red]",
                         "[red]file.md[/red]", "Ниже порога", "Cosine", "Передано модели: 0",
                         "использован исходный вопрос", "API не вернул статистику", "0.30 с"):
            self.assertIn(expected, rendered)

    def test_legacy_sources_remain_readable_without_new_diagnostics(self):
        output = io.StringIO()
        context = {"index": "legacy", "chunks": [{"source": "old.md", "section": "", "score": 0.8,
                                                    "chunk_id": "c1", "text": "Старый фрагмент"}]}
        with patch("llm_agent.cli.console", Console(file=output)):
            print_rag_sources(Turn(user="Вопрос", rag_enabled=True, rag_context=context))
        self.assertIn("old.md", output.getvalue())
        self.assertNotIn("Query rewrite", output.getvalue())


if __name__ == "__main__":
    unittest.main()
