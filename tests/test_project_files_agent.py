"""Существующий Agent + настоящий MCP HTTP; заменён только ответ LLM, без внешней сети."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openai.types.chat import ChatCompletion

from llm_agent.agent import Agent
from tests.helpers import completion
from tests.test_project_mcp_server import running_project_server


class ProjectFilesAgentTests(unittest.TestCase):
    def test_two_goals_use_mcp_results_create_report_and_edit_documentation(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "project"
            root.mkdir()
            sources = {
                "client.py": "class ApiClient:\n    timeout = 30\n    retries = 3\n",
                "service.py": "from client import ApiClient\nclient = ApiClient()\n",
                "worker.py": "from client import ApiClient\nworker_client = ApiClient()\n",
            }
            for name, content in sources.items():
                (root / name).write_text(content, encoding="utf-8")
            old = "# Клиент\n\nТаймаут: 5 секунд. Попытки: 1.\n\n## Запуск\npython service.py\n"
            new = old.replace("Таймаут: 5 секунд. Попытки: 1.", "Таймаут: 30 секунд. Попытки: 3.")
            (root / "README.md").write_text(old, encoding="utf-8")

            def invoke(request, name, arguments, call_id):
                definition = next(tool["function"] for tool in request["tools"]
                                  if tool["function"]["name"].startswith(f"mcp_{name}_"))
                response = completion().model_dump()
                response["choices"][0].update(finish_reason="tool_calls", message={
                    "role": "assistant", "content": None, "tool_calls": [{
                        "id": call_id, "type": "function", "function": {
                            "name": definition["name"], "arguments": json.dumps(arguments),
                        },
                    }],
                })
                return ChatCompletion.model_validate(response)

            def result_for(request, call_id):
                message = next(message for message in request["messages"]
                               if message.get("tool_call_id") == call_id)
                payload = json.loads(message["content"])
                self.assertFalse(payload["is_error"], payload)
                return payload["data"]

            def final(text):
                response = completion()
                response.choices[0].message.content = text
                return response

            report = "# Использования ApiClient\n\n" + "\n".join(f"- {name}" for name in sources) + "\n"
            step = 0

            def scripted_model(**request):
                nonlocal step
                step += 1
                if step == 1:
                    return invoke(request, "search_text", {"query": "ApiClient", "pattern": "*.py"}, "find")
                if step in (2, 3, 4):
                    matches = result_for(request, "find")["matches"]
                    # Пути берутся из настоящего ответа MCP, не передаются в цели пользователя.
                    paths = sorted({item["path"] for item in matches})
                    self.assertEqual(paths, sorted(sources))
                    return invoke(request, "read_file", {"path": paths[step - 2]}, f"read-{step}")
                if step == 5:
                    for read_id in ("read-2", "read-3", "read-4"):
                        self.assertIn("ApiClient", result_for(request, read_id)["content"])
                    return invoke(request, "create_file", {"path": "docs/api-usage.md", "content": report}, "create")
                if step == 6:
                    saved = result_for(request, "create")
                    self.assertTrue(saved["changed"])
                    self.assertIn("+", saved["diff"])
                    return final("Прочитаны три исходника; сохранён docs/api-usage.md.")
                if step == 7:
                    return invoke(request, "list_files", {}, "list-docs")
                if step == 8:
                    self.assertIn("README.md", result_for(request, "list-docs")["files"])
                    return invoke(request, "read_file", {"path": "client.py"}, "read-client")
                if step == 9:
                    self.assertIn("timeout = 30", result_for(request, "read-client")["content"])
                    return invoke(request, "read_file", {"path": "README.md"}, "read-doc")
                if step == 10:
                    document = result_for(request, "read-doc")
                    return invoke(request, "edit_file", {
                        "path": "README.md", "old_text": "Таймаут: 5 секунд. Попытки: 1.",
                        "new_text": "Таймаут: 30 секунд. Попытки: 3.",
                        "expected_sha256": document["sha256"],
                    }, "edit")
                if step == 11:
                    edited = result_for(request, "edit")
                    self.assertTrue(edited["changed"])
                    self.assertIn("+Таймаут: 30 секунд", edited["diff"])
                    return final("README.md обновлён по client.py; раздел запуска сохранён.")
                self.fail("Лишний запрос к модели")

            with running_project_server(root) as server:
                with patch("llm_agent.agent.OpenAI") as client:
                    client.return_value.chat.completions.create.side_effect = scripted_model
                    agent = Agent("test-only", parent / "history.json", mcp_servers=(server,))
                    try:
                        first = agent.request("Найди использования ApiClient и сохрани отчёт в документации проекта.")
                        second = agent.request("Обнови документацию по фактическим настройкам клиента.")
                    finally:
                        agent.close()
            self.assertEqual(step, 11)
            self.assertEqual((root / "docs/api-usage.md").read_text(encoding="utf-8"), report)
            self.assertEqual((root / "README.md").read_text(encoding="utf-8"), new)
            self.assertEqual([record.tool_name for record in first.tool_calls],
                             ["search_text", "read_file", "read_file", "read_file", "create_file"])
            self.assertEqual([record.tool_name for record in second.tool_calls],
                             ["list_files", "read_file", "read_file", "edit_file"])
            self.assertFalse(any(record.is_error for record in (*first.tool_calls, *second.tool_calls)))
            for name, content in sources.items():
                self.assertEqual((root / name).read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
