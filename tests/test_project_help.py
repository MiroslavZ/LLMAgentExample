"""Справка о проекте через настоящий сервис и интерфейсы, без сети и моделей."""

import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from nicegui import Client, core, ui
from openai.types.chat import ChatCompletion
from rich.console import Console

from llm_agent.cli import run
from llm_agent.history import HistoryManager
from llm_agent.indexing.chunking import load_corpus
from llm_agent.mcp_client import MCPConnectionError
from llm_agent.mcp_config import MCPServer
from llm_agent.models import ContextSettings, RequestOptions
from llm_agent.project_help import (
    DEFAULT_MCP_URL, DEFAULT_QUESTION, HELP_INSTRUCTION, ROOT,
    parse_help, validate_git,
)
from llm_agent.rag import RAGError
from llm_agent.rag_answer import unknown_answer
from llm_agent.service import ConversationService
from llm_agent.tool_events import ToolCallRecord
from llm_agent.web.components import render_turn
from llm_agent.web.jobs import RequestRunner
from llm_agent.web.page import ChatPage
from tests.helpers import completion, create_selected_conversation, rag_preparation


DOCUMENT = "Основное ядро агента находится в llm_agent/agent.py."


def project_context():
    return {"index": "shared.sqlite3", "chunks": [{
        "chunk_id": "structure-1", "source": "docs/PROJECT_STRUCTURE.md",
        "section": "Основные модули", "text": DOCUMENT, "score": 0.91,
    }]}


def document_answer():
    return {
        "status": "answered", "answer": DOCUMENT, "sources": ["structure-1"],
        "quotes": [{"chunk_id": "structure-1", "text": DOCUMENT}], "clarification": None,
    }


def json_completion(payload):
    response = completion()
    response.choices[0].message.content = json.dumps(payload, ensure_ascii=False)
    return response


def document_completion(payload=None, *, wrap=True):
    payload = document_answer() if payload is None else payload
    return json_completion({"documentation": payload} if wrap else payload)


def tool_completion(name):
    payload = completion().model_dump()
    payload["choices"][0].update(
        finish_reason="tool_calls",
        message={"role": "assistant", "content": None, "tool_calls": [{
            "id": "selected-git", "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }]},
    )
    return ChatCompletion.model_validate(payload)


def git_payload(**changes):
    return {
        "project_root": str(ROOT), "branch": "feature/day31", "commit": "a" * 40,
        "detached": False, **changes,
    }


def mcp_result(payload=None, *, is_error=False):
    return SimpleNamespace(is_error=is_error, structured_content=payload or git_payload(), content=[])


class ProjectHelpParsingTests(unittest.TestCase):
    def test_command_requires_exact_name_and_preserves_question(self):
        for text, expected in (
            ("/help", DEFAULT_QUESTION), (" \t/help\n ", DEFAULT_QUESTION),
            ("/help Где ядро?", "Где ядро?"),
            ("/help\nКак устроен проект?\nГде тесты?", "Как устроен проект?\nГде тесты?"),
            ("/helpful", None), ("/help/README", None), ("/HELP", None),
            ("Объясни /help", None), ("", None),
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_help(text), expected)

    def test_git_validation_rejects_other_repository_and_inconsistent_state(self):
        for payload in (
            git_payload(project_root=str(ROOT.parent)),
            git_payload(branch=None, detached=False),
            git_payload(branch="main", detached=True),
            git_payload(branch=None, commit=None, detached=True),
            git_payload(branch="bad\nbranch"), git_payload(commit="not-a-commit"),
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_git(payload, str(ROOT))
        self.assertEqual(validate_git(git_payload(commit=None), str(ROOT))["branch"], "feature/day31")
        self.assertTrue(validate_git(git_payload(branch=None, detached=True), str(ROOT))["detached"])



class ProjectHelpFixture:
    def prepare_service(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.enterContext(patch.dict(os.environ, {
            "PROJECT_HELP_MCP_URL": DEFAULT_MCP_URL,
        }))
        self.openai = self.enterContext(patch("llm_agent.agent.OpenAI"))
        self.complete = self.openai.return_value.chat.completions.create
        self.complete.return_value = document_completion()
        self.call_tool = self.enterContext(patch("llm_agent.mcp_tools.call_tool", new_callable=AsyncMock))
        self.call_tool.return_value = mcp_result()
        self.get_tools = self.enterContext(patch(
            "llm_agent.mcp_tools.get_tools", new_callable=AsyncMock,
            side_effect=AssertionError("Справка не должна подключаться к MCP до выбора модели"),
        ))
        self.service = ConversationService(self.directory)
        self.conversation = create_selected_conversation(self.service)
        self.server = MCPServer("project", "Текущий проект", DEFAULT_MCP_URL, enabled=True)
        self.service.mcp_servers.save(self.server)
        self.service.retriever = Mock()
        self.service.retriever.retrieve.return_value = project_context()

    def send_help(self, text="/help", **kwargs):
        return self.service.send(self.conversation.id, text, **kwargs)

    def select_git(self, final=None, *, before_final=None):
        """Модель выбирает реальную функцию из переданного ей каталога."""
        selected = False

        def answer(**kwargs):
            nonlocal selected
            if not selected:
                selected = True
                functions = [tool["function"] for tool in kwargs["tools"]]
                function = next(function for function in functions if "git_branch" in function["name"])
                return tool_completion(function["name"])
            if before_final is not None:
                before_final()
            if isinstance(final, Exception):
                raise final
            return final if final is not None else json_completion({"documentation": None})

        self.complete.side_effect = answer


class ProjectHelpCorpusTests(unittest.TestCase):
    def test_existing_manifests_cover_project_documentation_without_course_materials(self):
        expected = {"README.md"} | {
            path.relative_to(ROOT).as_posix() for path in (ROOT / "docs").glob("*.md")
            if not path.name.startswith("DAY")
            and path.name not in {"CHAT.md", "TASKS.md", "SOLUTIONS.md", "LECTURE.md"}
        }
        for manifest in ("examples/indexing/corpus.json", "my-corpus.json"):
            with self.subTest(manifest=manifest):
                documents = load_corpus(ROOT, ROOT / manifest)
                self.assertEqual({document.source for document in documents}, expected)
                self.assertEqual(len(documents), len(expected))


class ProjectHelpServiceTests(ProjectHelpFixture, unittest.TestCase):
    def setUp(self):
        self.prepare_service()

    def test_help_and_ordinary_rag_use_same_configured_retriever(self):
        with patch.dict(os.environ, {"RAG_INDEX_PATH": "data/custom/structural.sqlite3"}):
            self.service = ConversationService(self.directory)
        retriever = self.service.retriever
        self.assertEqual(retriever.path, ROOT / "data/custom/structural.sqlite3")
        context = dict(project_context(), index=str(retriever.path))
        with patch.object(retriever, "retrieve", return_value=context) as search, \
                patch("llm_agent.service.prepare_turn", side_effect=rag_preparation):
            first = self.send_help("/help Где ядро?")
            self.service.mcp_servers.save(replace(self.server, enabled=False))
            self.complete.return_value = document_completion(wrap=False)
            second = self.service.send(first.id, "Как устроен проект?", settings=ContextSettings(rag_enabled=True))
        self.assertEqual(first.turns[-1].status, "completed")
        self.assertEqual(second.turns[-1].status, "completed")
        self.assertIs(self.service.retriever, retriever)
        self.assertEqual([call.args[0] for call in search.call_args_list], ["Где ядро?", "Как устроен проект?"])
        self.assertEqual(first.turns[-1].rag_context["index"], str(retriever.path))
        self.assertEqual(second.turns[-1].rag_context["index"], str(retriever.path))

    def test_default_help_uses_shared_rag_without_changing_rag_setting(self):
        result = self.send_help()
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertTrue(result.turns[-1].rag_enabled)
        self.assertFalse(result.settings.rag_enabled)
        self.service.retriever.retrieve.assert_called_once()
        self.assertEqual(self.service.retriever.retrieve.call_args.args[0], DEFAULT_QUESTION)
        self.assertEqual(result.turns[-1].project_help["question"], DEFAULT_QUESTION)
        self.assertIn(DOCUMENT, result.turns[-1].answer)
        self.assertNotIn("Git", result.turns[-1].answer)
        self.assertIsNone(result.turns[-1].project_help["git"])
        self.assertIsNone(result.turns[-1].project_help["git_error"])
        self.assertEqual(result.turns[-1].tool_calls, [])
        self.call_tool.assert_not_awaited()
        self.get_tools.assert_not_awaited()
        self.assertEqual(self.complete.call_args.kwargs["tool_choice"], "auto")

    def test_enabled_rewrite_prepares_original_message_with_existing_dialogue(self):
        self.send_help("/help git branch")
        self.service.update_settings(self.conversation.id, ContextSettings(rag_rewrite_enabled=True))
        rewrite = {
            "query": "Где находится ядро агента в проекте?", "status": "success",
            "reason": None, "elapsed_seconds": 0.0, "usage": None,
        }

        def prepare(*args, **kwargs):
            return dict(rag_preparation(*args, **kwargs), rewrite=rewrite)

        with patch("llm_agent.service.prepare_turn", side_effect=prepare) as preparation:
            result = self.send_help("/help   А где ядро?  ")
        preparation.assert_called_once()
        self.assertEqual(preparation.call_args.args[0], "/help   А где ядро?")
        self.assertEqual(preparation.call_args.args[2][0].user, "/help git branch")
        self.assertEqual(preparation.call_args.args[2][-1].user, "/help   А где ядро?")
        self.assertTrue(preparation.call_args.kwargs["rewrite_enabled"])
        self.assertEqual(self.service.retriever.retrieve.call_args.kwargs["rewrite"], rewrite)
        self.assertEqual(result.turns[-1].rag_preparation["status"], "success")
        self.assertEqual(result.turns[-1].status, "completed")

    def test_real_preparation_succeeds_and_memory_quotes_original_help_message(self):
        self.send_help("/help Где находится ядро?")
        self.service.update_settings(self.conversation.id, ContextSettings(rag_rewrite_enabled=True))
        question = "/help   Нужен только CLI. А где ядро?"
        response = completion()
        response.choices[0].message.content = json.dumps({
            "memory_delta": [{
                "op": "set", "field": "constraints", "key": "interface",
                "value": "CLI", "quote": "только CLI",
            }],
            "standalone_query": "/help Где находится ядро агента для CLI?",
        }, ensure_ascii=False)
        with patch("llm_agent.rag_dialogue.OpenAI") as preparation_api:
            create = preparation_api.return_value.__enter__.return_value.chat.completions.create
            create.return_value = response
            result = self.send_help(question)
        create.assert_called_once()
        payload = json.loads(create.call_args.kwargs["messages"][-1]["content"])
        self.assertEqual(payload["current_user"], question)
        self.assertEqual(payload["recent_turns"][0]["user"], "/help Где находится ядро?")
        self.assertTrue(payload["rewrite_enabled"])
        self.assertEqual(result.turns[-1].rag_preparation["status"], "success")
        entry = result.dialogue_task_memory["constraints"]["interface"]
        self.assertEqual(entry, {"value": "CLI", "quote": "только CLI", "turn": 2})
        self.assertIn(entry["quote"], result.turns[entry["turn"] - 1].user)
        search_call = self.service.retriever.retrieve.call_args
        self.assertEqual(search_call.args[0], "Нужен только CLI. А где ядро?")
        self.assertEqual(search_call.kwargs["rewrite"]["status"], "success")
        self.assertEqual(search_call.kwargs["rewrite"]["query"], "Где находится ядро агента для CLI?")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(ConversationService(self.directory).get(result.id), result)

    def test_real_preparation_failure_searches_question_without_command_prefix(self):
        self.service.update_settings(self.conversation.id, ContextSettings(rag_rewrite_enabled=True))
        before = self.service.get(self.conversation.id).dialogue_task_memory
        with patch("llm_agent.rag_dialogue.OpenAI") as preparation_api:
            create = preparation_api.return_value.__enter__.return_value.chat.completions.create
            create.side_effect = RuntimeError("private-provider-error")
            result = self.send_help("/help Где ядро?")
        create.assert_called_once()
        rewrite = self.service.retriever.retrieve.call_args.kwargs["rewrite"]
        self.assertEqual(rewrite["status"], "fallback")
        self.assertEqual(rewrite["query"], "Где ядро?")
        self.assertEqual(result.dialogue_task_memory, before)
        self.assertEqual(result.turns[-1].rag_preparation["status"], "fallback")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertNotIn("private-provider-error", str(result.turns[-1]))

    def test_bare_help_with_rewrite_preserves_memory_without_preparing_synthetic_question(self):
        previous = self.send_help("/help Нужен только CLI. Где ядро?")
        previous.dialogue_task_memory["constraints"]["interface"] = {
            "value": "CLI", "quote": "только CLI", "turn": 1,
        }
        self.service.store.save(previous)
        before = deepcopy(previous.dialogue_task_memory)
        self.service.update_settings(self.conversation.id, ContextSettings(rag_rewrite_enabled=True))
        with patch("llm_agent.rag_dialogue.OpenAI") as preparation_api:
            result = self.send_help("  /help  ")
        preparation_api.assert_not_called()
        search_call = self.service.retriever.retrieve.call_args
        self.assertEqual(search_call.args[0], DEFAULT_QUESTION)
        self.assertEqual(search_call.kwargs["rewrite"]["query"], DEFAULT_QUESTION)
        self.assertEqual(search_call.kwargs["rewrite"]["status"], "fallback")
        self.assertEqual(result.dialogue_task_memory, before)
        self.assertEqual(result.turns[-1].user, "/help")
        self.assertIsNone(result.turns[-1].rag_preparation)
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(ConversationService(self.directory).get(result.id), result)

    def test_git_runs_only_after_model_selects_tool_from_project_connection(self):
        self.select_git()
        result = self.send_help("/help Какая сейчас ветка?")
        self.call_tool.assert_awaited_once_with(self.server, "git_branch", {})
        turn = result.turns[-1]
        self.assertEqual(turn.project_help["git"], git_payload())
        self.assertEqual(turn.tool_calls[0].tool_name, "git_branch")
        self.assertEqual(json.loads(turn.tool_calls[0].result)["data"], git_payload())
        self.assertIn("Источник Git: MCP / git_branch", turn.answer)
        self.assertEqual(turn.status, "completed")
        self.assertIsNone(turn.rag_answer)
        self.assertEqual(self.complete.call_count, 2)
        self.get_tools.assert_not_awaited()
        self.assertEqual(ConversationService(self.directory).get(result.id), result)

    def test_branch_question_does_not_force_tool_without_model_selection(self):
        self.complete.return_value = document_completion(unknown_answer())
        result = self.send_help("/help Какая сейчас ветка?")
        self.call_tool.assert_not_awaited()
        self.assertEqual(result.turns[-1].tool_calls, [])
        self.assertIsNone(result.turns[-1].project_help["git"])
        self.assertIn("Не знаю", result.turns[-1].answer)
        self.assertNotIn("feature/day31", result.turns[-1].answer)

    def test_current_commit_question_returns_sha_from_model_selected_tool(self):
        self.select_git()
        result = self.send_help("/help Какой текущий коммит?")
        turn = result.turns[-1]
        self.assertEqual(turn.status, "completed")
        self.assertIn("a" * 40, turn.answer)
        self.assertEqual(turn.project_help["git"]["commit"], "a" * 40)
        self.assertIsNone(turn.rag_answer)
        self.call_tool.assert_awaited_once_with(self.server, "git_branch", {})

    def test_null_documentation_without_tool_result_is_rejected_and_repaired(self):
        self.complete.side_effect = [json_completion({"documentation": None}), document_completion()]
        result = self.send_help("/help Где находится ядро?")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIn(DOCUMENT, result.turns[-1].answer)
        self.assertEqual(self.complete.call_count, 2)
        self.assertEqual(self.complete.call_args.kwargs["tool_choice"], "none")
        self.call_tool.assert_not_awaited()

    def test_invalid_document_quote_is_repaired_without_repeating_tool(self):
        bad = document_answer()
        bad["quotes"][0]["text"] = "Несуществующая цитата"
        finals = iter([document_completion(bad), document_completion()])
        selected = False

        def answer(**kwargs):
            nonlocal selected
            if not selected:
                selected = True
                function = next(tool["function"] for tool in kwargs["tools"]
                                if "git_branch" in tool["function"]["name"])
                return tool_completion(function["name"])
            return next(finals)

        self.complete.side_effect = answer
        result = self.send_help("/help Какая ветка и где ядро?")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(result.turns[-1].rag_answer["quotes"], [{"chunk_id": "structure-1", "text": DOCUMENT}])
        self.assertIn("feature/day31", result.turns[-1].answer)
        self.assertEqual(self.complete.call_count, 3)
        self.assertEqual(self.complete.call_args.kwargs["tool_choice"], "none")
        self.call_tool.assert_awaited_once()

    def test_disabled_mcp_connection_never_executes_tool_but_documents_work(self):
        self.service.mcp_servers.save(replace(self.server, enabled=False))
        result = self.send_help()
        self.call_tool.assert_not_awaited()
        self.assertIsNone(result.turns[-1].project_help["git"])
        self.assertIsNone(result.turns[-1].project_help["git_error"])
        self.assertIn(DOCUMENT, result.turns[-1].answer)
        self.assertEqual(result.turns[-1].tool_calls, [])
        self.assertNotIn("tools", self.complete.call_args.kwargs)

    def test_missing_project_connection_does_not_block_documentation(self):
        self.service.mcp_servers.delete(self.server.id)
        result = self.send_help()
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIn(DOCUMENT, result.turns[-1].answer)
        self.assertEqual(result.turns[-1].tool_calls, [])
        self.assertNotIn("tools", self.complete.call_args.kwargs)
        self.call_tool.assert_not_awaited()

    def test_disabled_connection_cannot_supply_branch_and_creates_no_fake_call(self):
        self.service.mcp_servers.save(replace(self.server, enabled=False))
        self.complete.return_value = document_completion(unknown_answer())
        result = self.send_help("/help Какая сейчас ветка?")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIn("Не знаю", result.turns[-1].answer)
        self.assertNotIn("feature/day31", result.turns[-1].answer)
        self.assertEqual(result.turns[-1].tool_calls, [])
        self.assertNotIn("tools", self.complete.call_args.kwargs)
        self.call_tool.assert_not_awaited()

    def test_mcp_from_different_repository_is_not_presented_as_current_branch(self):
        self.call_tool.return_value = mcp_result(git_payload(project_root=str(ROOT.parent), branch="other"))
        self.select_git()
        result = self.send_help("/help git branch")
        self.assertIsNone(result.turns[-1].project_help["git"])
        self.assertIn("другой репозиторий", result.turns[-1].answer)
        self.assertNotIn("`other`", result.turns[-1].answer)
        self.assertTrue(result.turns[-1].tool_calls[0].is_error)

    def test_text_only_mcp_result_and_detached_head_are_supported(self):
        payload = git_payload(branch=None, detached=True)
        self.call_tool.return_value = SimpleNamespace(
            is_error=False, structured_content=None,
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        )
        self.select_git()
        result = self.send_help("/help git branch")
        self.assertIn("detached HEAD", result.turns[-1].answer)
        self.assertIn("a" * 40, result.turns[-1].answer)
        self.assertEqual(self.complete.call_count, 2)

    def test_mixed_answer_keeps_document_sources_separate_from_git(self):
        seen_at_request = []

        def before_final():
            seen_at_request.append(self.service.store.load(self.conversation.id).turns[-1])

        self.select_git(document_completion(), before_final=before_final)
        result = self.send_help("/help Какая ветка и где находится ядро?")
        turn = result.turns[-1]
        self.assertEqual([source["source"] for source in turn.rag_answer["sources"]],
                         ["docs/PROJECT_STRUCTURE.md"])
        self.assertEqual(turn.rag_answer["quotes"], [{"chunk_id": "structure-1", "text": DOCUMENT}])
        self.assertIn("feature/day31", turn.answer)
        self.assertEqual(seen_at_request[0].project_help["git"], git_payload())
        self.assertEqual(seen_at_request[0].rag_context, project_context())
        messages = self.complete.call_args.kwargs["messages"]
        self.assertTrue(any(item["role"] == "system" and HELP_INSTRUCTION in item["content"]
                            for item in messages))
        self.assertTrue(any(item["role"] == "tool" and "feature/day31" in item["content"]
                            for item in messages))
        self.assertEqual(len(seen_at_request[0].tool_calls), 1)
        self.assertEqual(ConversationService(self.directory).get(result.id), result)

    def test_unavailable_mcp_still_returns_document_answer(self):
        self.call_tool.side_effect = MCPConnectionError("MCP-сервер недоступен")
        self.select_git(document_completion())
        result = self.send_help("/help Какая ветка и где находится ядро?")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIn(DOCUMENT, result.turns[-1].answer)
        self.assertIn("MCP-сервер недоступен", result.turns[-1].answer)
        self.assertIsNone(result.turns[-1].project_help["git"])
        self.assertTrue(result.turns[-1].tool_calls[0].is_error)

    def test_missing_index_still_allows_model_to_select_git(self):
        self.service.retriever.retrieve.side_effect = RAGError("missing local index")
        self.select_git()
        result = self.send_help("/help Какая сейчас ветка?")
        turn = result.turns[-1]
        self.assertEqual(turn.status, "completed")
        self.assertIn("feature/day31", turn.answer)
        self.assertIn("Пересоберите индекс", turn.answer)
        self.assertIn("docs/PROJECT_HELP.md", turn.answer)
        self.assertIsNone(turn.rag_answer)
        self.assertEqual(self.complete.call_count, 2)
        self.call_tool.assert_awaited_once()
        self.assertEqual(ConversationService(self.directory).get(result.id), result)

    def test_empty_documents_still_allow_model_to_select_git(self):
        self.service.retriever.retrieve.return_value = dict(project_context(), chunks=[])
        self.select_git()
        result = self.send_help("/help Какая сейчас ветка?")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIn("feature/day31", result.turns[-1].answer)
        self.assertIsNone(result.turns[-1].rag_answer)
        self.call_tool.assert_awaited_once()

    def test_weak_documents_cannot_prove_answer_but_do_not_block_git(self):
        context = project_context()
        context["chunks"][0]["score"] = 0.01
        self.service.retriever.retrieve.return_value = context
        self.select_git()
        result = self.send_help("/help Какая сейчас ветка?")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIn("feature/day31", result.turns[-1].answer)
        self.call_tool.assert_awaited_once()
        messages = self.complete.call_args.kwargs["messages"]
        self.assertFalse(any(DOCUMENT in (item.get("content") or "") for item in messages))

    def test_unsupported_question_returns_unknown_documentation_without_git(self):
        self.complete.return_value = document_completion(unknown_answer())
        result = self.send_help("/help Какая SLA у будущего production deployment?")
        turn = result.turns[-1]
        self.assertEqual(turn.status, "completed")
        self.assertEqual(turn.rag_answer["status"], "unknown")
        self.assertEqual(turn.rag_answer["sources"], [])
        self.assertIn("Не знаю", turn.answer)
        self.assertNotIn("feature/day31", turn.answer)
        self.call_tool.assert_not_awaited()

    def test_model_failure_before_selection_creates_no_git_call_record(self):
        self.complete.side_effect = RuntimeError("secret-provider-detail")
        result = self.send_help()
        turn = result.turns[-1]
        self.assertEqual(turn.status, "error")
        self.assertIsNone(turn.project_help["git"])
        self.assertEqual(turn.tool_calls, [])
        self.assertNotIn("secret-provider-detail", turn.error)
        self.call_tool.assert_not_awaited()
        self.assertEqual(ConversationService(self.directory).get(result.id), result)

    def test_model_failure_after_tool_preserves_git_record_and_documents_on_disk(self):
        self.select_git(RuntimeError("secret-provider-detail"))
        result = self.send_help("/help Какая ветка и где ядро?")
        turn = result.turns[-1]
        self.assertEqual(turn.status, "partial")
        self.assertEqual(turn.project_help["git"], git_payload())
        self.assertEqual(turn.rag_context, project_context())
        self.assertEqual(json.loads(turn.tool_calls[0].result)["data"], git_payload())
        self.assertNotIn("secret-provider-detail", turn.error)
        self.assertEqual(ConversationService(self.directory).get(result.id), result)

    def test_help_round_trip_preserves_settings_memory_and_paused_task(self):
        settings = ContextSettings(strategy="window", window_size=2, rag_enabled=False)
        self.service.update_settings(self.conversation.id, settings)
        self.service.start_task(self.conversation.id, "Подготовить функцию")
        before = self.service.pause_task(self.conversation.id)
        result = self.send_help("/help Где находится ядро?")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(result.task_state, before.task_state)
        self.assertEqual(result.settings, before.settings)
        self.assertEqual(result.dialogue_task_memory, before.dialogue_task_memory)
        self.assertEqual(ConversationService(self.directory).get(result.id), result)
        history = HistoryManager._decode_data(result.working_context)[0]
        self.assertEqual(history[-2]["content"], "/help Где находится ядро?")
        self.assertEqual(history[-1]["content"], result.turns[-1].answer)

    def test_meta_prompt_option_does_not_add_second_request_or_change_command(self):
        result = self.send_help(options=RequestOptions(meta_prompt=True))
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIsNone(result.turns[-1].meta_prompt)
        self.assertEqual(self.complete.call_count, 1)
        self.assertIn(DOCUMENT, result.turns[-1].answer)
        history = HistoryManager._decode_data(result.working_context)[0]
        self.assertEqual(history[-2]["content"], "/help")

    def test_disabled_model_tools_do_not_call_git_and_documents_still_work(self):
        model = self.service.get_model(self.conversation.id)
        self.service.models.save(replace(model, tools_enabled=False))
        result = self.send_help()
        self.call_tool.assert_not_awaited()
        self.assertIn(DOCUMENT, result.turns[-1].answer)
        self.assertNotIn("feature/day31", result.turns[-1].answer)
        self.assertEqual(result.turns[-1].tool_calls, [])
        self.assertNotIn("tools", self.complete.call_args.kwargs)

    def test_disabled_model_tools_cannot_answer_current_branch_from_history(self):
        self.select_git()
        self.send_help("/help Какая сейчас ветка?")
        model = self.service.get_model(self.conversation.id)
        self.service.models.save(replace(model, tools_enabled=False))
        self.complete.side_effect = None
        self.complete.return_value = document_completion(unknown_answer())
        self.call_tool.reset_mock()
        result = self.send_help("/help Какая сейчас ветка?")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIsNone(result.turns[-1].project_help["git"])
        self.assertEqual(result.turns[-1].tool_calls, [])
        self.assertNotIn("feature/day31", result.turns[-1].answer)
        self.assertNotIn("tools", self.complete.call_args.kwargs)
        self.call_tool.assert_not_awaited()

    def test_ordinary_message_does_not_activate_project_help(self):
        self.service.mcp_servers.save(replace(self.server, enabled=False))
        self.complete.return_value = completion()
        result = self.service.send(self.conversation.id, "Объясни /helpful")
        self.call_tool.assert_not_awaited()
        self.service.retriever.retrieve.assert_not_called()
        self.assertIsNone(result.turns[-1].project_help)
        self.assertFalse(result.turns[-1].rag_enabled)
        self.assertEqual(result.turns[-1].answer, "Ответ")

    def test_json_response_keeps_git_and_document_sources_in_separate_fields(self):
        self.select_git(document_completion())
        result = self.send_help("/help Какая ветка и где ядро?", options=RequestOptions(response_format="object"))
        payload = json.loads(result.turns[-1].answer)
        self.assertEqual(payload["project"]["git"], git_payload())
        self.assertEqual(payload["documentation"]["sources"][0]["source"], "docs/PROJECT_STRUCTURE.md")
        self.assertEqual(ConversationService(self.directory).get(result.id), result)

    def test_legacy_help_history_with_programmatic_git_and_rag_still_loads(self):
        result = self.send_help("/help Где ядро?")
        turn = result.turns[-1]
        turn.project_help["git"] = git_payload()
        turn.tool_calls = [ToolCallRecord(
            call_id="legacy-git", server_name=self.server.name, tool_name="git_branch",
            arguments="{}", result=json.dumps(git_payload()), is_error=False, elapsed_seconds=0.01,
        )]
        turn.answer += (
            "\n\nТекущая Git-ветка: `feature/day31`."
            f"\n\nИсточник Git: MCP / git_branch · {ROOT}"
        )
        self.assertNotIn("a" * 40, turn.answer)
        self.service.store.save(result)
        restored = ConversationService(self.directory).get(result.id)
        self.assertEqual(restored.turns[-1].project_help, turn.project_help)
        self.assertEqual(restored.turns[-1].rag_answer, turn.rag_answer)
        self.assertEqual(restored.turns[-1].tool_calls, turn.tool_calls)
        self.assertIn("a" * 40, restored.turns[-1].answer)
        self.assertEqual(restored.turns[-1].rag_answer["quotes"], [{"chunk_id": "structure-1", "text": DOCUMENT}])
        self.assertEqual(json.loads(restored.turns[-1].tool_calls[0].result), git_payload())


class ProjectHelpCLITests(ProjectHelpFixture, unittest.TestCase):
    def setUp(self):
        self.prepare_service()
        self.enterContext(patch("llm_agent.cli.load_env"))
        # Настоящий CLI создаёт новый сервис для того же каталога. Общий mock
        # подменяет только тяжёлый поиск по индексу, не маршрутизацию команды.
        self.enterContext(patch("llm_agent.service.Retriever", return_value=self.service.retriever))

    def cli(self, *args):
        output = io.StringIO()
        argv = ["main.py", "--conversation", str(self.service.store.path(self.conversation.id)), *args]
        with patch("sys.argv", argv), patch("llm_agent.cli.console", Console(file=output, width=240)), \
                contextlib.redirect_stdout(output):
            run()
        return output.getvalue()

    def test_help_question_through_user_flag_uses_same_service_and_persists_sources(self):
        output = self.cli("--user", "/help Где ядро?")
        self.assertIn(DOCUMENT, output)
        self.assertNotIn("feature/day31", output)
        saved = self.service.get(self.conversation.id)
        self.assertEqual(saved.turns[-1].user, "/help Где ядро?")
        self.assertIsNone(saved.turns[-1].project_help["git"])
        self.assertFalse(saved.settings.rag_enabled)
        self.call_tool.assert_not_awaited()

    def test_batch_branch_request_uses_model_selected_tool_and_returns_valid_json(self):
        self.select_git()
        payload = json.loads(self.cli("--batch", "--user", "/help Какая сейчас ветка?"))
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["project_help"]["git"], git_payload())
        self.assertIn("feature/day31", payload["answer"])
        self.assertEqual(self.complete.call_count, 2)
        self.call_tool.assert_awaited_once()


class ProjectHelpWebTests(ProjectHelpFixture, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.prepare_service()
        self.enterContext(patch("nicegui.background_tasks.create_or_defer",
                                side_effect=lambda coroutine, **_: coroutine.close()))
        self.enterContext(patch("llm_agent.web.page.ui.timer"))
        self.enterContext(patch("llm_agent.web.page.ui.navigate.history.replace"))
        self.notify = self.enterContext(patch("llm_agent.web.page.ui.notify"))
        self.enterContext(patch.object(core, "loop", asyncio.get_running_loop()))
        self.client = Client(ui.page("/project-help-unit-test"))
        self.addCleanup(self.client.delete)
        self.runner = RequestRunner(self.service)
        self.enterContext(patch("llm_agent.web.jobs.run.io_bound", side_effect=asyncio.to_thread))

    async def submit_from_page(self, question):
        with self.client:
            page = ChatPage(self.service, self.runner, conversation_id=self.conversation.id)
            page.build()
            page.user_input.set_value(question)
            page.send()
        self.assertIn(self.conversation.id, self.runner._tasks)
        task = self.runner._tasks[self.conversation.id]
        await asyncio.wait_for(task, timeout=5)
        self.assertEqual(self.runner.errors, {})
        return self.service.get(self.conversation.id)

    async def test_web_question_reaches_shared_help_path_with_rag_switch_off(self):
        result = await self.submit_from_page("/help Где ядро?")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIn(DOCUMENT, result.turns[-1].answer)
        self.assertNotIn("feature/day31", result.turns[-1].answer)
        self.assertFalse(result.settings.rag_enabled)
        self.call_tool.assert_not_awaited()
        with self.client, ui.column() as transcript:
            render_turn(result.turns[-1], restore=lambda: None, busy=False)
        markdown = [element.content for element in transcript.descendants() if isinstance(element, ui.markdown)]
        self.assertIn(result.turns[-1].answer, markdown)
        self.assertTrue(any(DOCUMENT in text and "docs/PROJECT_STRUCTURE.md" in text for text in markdown))

    async def test_paused_web_composer_enables_send_only_for_exact_help_command(self):
        self.service.start_task(self.conversation.id, "Подготовить функцию")
        self.service.pause_task(self.conversation.id)
        with self.client:
            page = ChatPage(self.service, self.runner, conversation_id=self.conversation.id)
            page.build()
            self.assertFalse(page.send_button.enabled)
            page.user_input.set_value("/help Где ядро?")
            self.assertTrue(page.send_button.enabled)
            page.user_input.set_value("/helpful")
            self.assertFalse(page.send_button.enabled)
            page.user_input.set_value("Продолжи задачу")
            self.assertFalse(page.send_button.enabled)

    async def test_paused_task_allows_help_from_web_without_resuming_task(self):
        self.service.start_task(self.conversation.id, "Подготовить функцию")
        before = self.service.pause_task(self.conversation.id)
        self.select_git()
        result = await self.submit_from_page("/help git branch")
        self.assertEqual(result.task_state, before.task_state)
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIn("feature/day31", result.turns[-1].answer)
        self.assertEqual(self.complete.call_count, 2)
        self.call_tool.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
