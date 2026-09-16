"""Проверка реальной точки запуска HTTP-сервера без браузера и запросов к LLM."""

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from llm_agent.service import ConversationService


class WebStartupTests(unittest.TestCase):
    def test_module_serves_chat_and_reopens_selected_conversation(self):
        project_root = Path(__file__).resolve().parents[1]
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        # Пустое явное значение также предотвращает чтение настоящего ключа из .env.
        environment = {**os.environ, "API_KEY": "", "PYTHONIOENCODING": "utf-8"}
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                [sys.executable, "-m", "llm_agent.web", "--port", str(port),
                 "--data-dir", directory, "--no-browser"],
                cwd=project_root, env=environment, stdout=output, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=3) as client:
                    deadline = time.monotonic() + 20
                    while True:
                        if process.poll() is not None:
                            output.seek(0)
                            self.fail(output.read().decode("utf-8", errors="replace"))
                        try:
                            response = client.get("/")
                            break
                        except httpx.ConnectError:
                            if time.monotonic() >= deadline:
                                self.fail("Веб-сервер не запустился за 20 секунд")
                            time.sleep(0.1)
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("chat-app", response.text)
                    self.assertIn("Системный промпт", response.text)
                    self.assertIn("Ваше сообщение", response.text)
                    self.assertIn("API_KEY", response.text)
                    files = list(Path(directory).glob("*.json"))
                    self.assertEqual(len(files), 1)
                    storage = ConversationService(Path(directory), token=None)
                    saved = storage.get(files[0].stem)
                    saved.title = "Диалог из хранилища"
                    storage.store.save(saved)
                    reopened = client.get("/", params={"conversation": files[0].stem})
                    self.assertEqual(reopened.status_code, 200)
                    self.assertIn("Диалог из хранилища", reopened.text)
                    self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
