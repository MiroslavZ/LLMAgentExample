import io
import unittest
from unittest.mock import patch

from openai.types.chat import ChatCompletion
from rich.console import Console

import main
from agent import BASE_URL, DEFAULT_MODEL, META_PROMPT_SYSTEM, Agent, RequestResult


def make_response(content: str | None) -> ChatCompletion:
    return ChatCompletion(
        id="test-response",
        created=0,
        model=DEFAULT_MODEL,
        object="chat.completion",
        choices=[{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
    )


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        client_patch = patch("agent.OpenAI")
        self.client_factory = client_patch.start()
        self.addCleanup(client_patch.stop)
        self.create = self.client_factory.return_value.chat.completions.create
        self.create.return_value = make_response("Answer")
        self.agent = Agent("test-token")

    def test_request_returns_response_and_elapsed_time(self) -> None:
        with patch("agent.time.perf_counter", side_effect=[10.0, 12.5]):
            result = self.agent.request("Question")

        self.client_factory.assert_called_once_with(api_key="test-token", base_url=BASE_URL)
        self.create.assert_called_once_with(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Question"}],
            response_format={"type": "text"},
        )
        self.assertIs(result.response, self.create.return_value)
        self.assertEqual(result.content, "Answer")
        self.assertEqual(result.elapsed, 2.5)

    def test_request_options_do_not_leak_into_next_call(self) -> None:
        self.agent.request(
            "Question", model="custom-model", system="System", max_tokens=100,
            temperature=0.0, stop_sequences=["END"], response_format="object",
        )
        self.create.assert_called_with(
            model="custom-model",
            messages=[
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Question"},
            ],
            max_tokens=100, temperature=0.0, stop=["END"],
            response_format={"type": "json_object"},
        )

        self.agent.request("Next", stop_sequences=[])
        self.create.assert_called_with(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Next"}],
            response_format={"type": "text"},
        )

    def test_meta_prompt_uses_generated_text_and_separate_formats(self) -> None:
        for response_format, api_format in (
            ("text", "text"), ("object", "json_object"), ("schema", "json_schema"),
        ):
            with self.subTest(response_format=response_format):
                self.create.reset_mock()
                meta_response = make_response("Optimized question")
                final_response = make_response("Final answer")
                self.create.side_effect = [meta_response, final_response]

                meta_result, result = self.agent.request_with_meta_prompt(
                    "Question", model="custom-model", system="System",
                    max_tokens=100, temperature=0.0, stop_sequences=["END"],
                    response_format=response_format,
                )

                self.assertEqual(self.create.call_count, 2)
                first, second = [call.kwargs for call in self.create.call_args_list]
                for request in (first, second):
                    self.assertEqual(request["model"], "custom-model")
                    self.assertEqual(request["max_tokens"], 100)
                    self.assertEqual(request["temperature"], 0.0)
                    self.assertEqual(request["stop"], ["END"])
                self.assertEqual(first["response_format"], {"type": "text"})
                self.assertEqual(first["messages"], [
                    {"role": "system", "content": META_PROMPT_SYSTEM},
                    {"role": "user", "content": "Question"},
                ])
                self.assertEqual(second["response_format"], {"type": api_format})
                self.assertEqual(second["messages"], [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Optimized question"},
                ])
                self.assertIs(meta_result.response, meta_response)
                self.assertIs(result.response, final_response)

    def test_missing_content_returns_empty_string(self) -> None:
        self.create.return_value = make_response(None)
        self.assertEqual(self.agent.request("Question").content, "")

    def test_meta_prompt_stops_after_api_error(self) -> None:
        self.create.side_effect = RuntimeError("API unavailable")
        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            self.agent.request_with_meta_prompt("Question")
        self.create.assert_called_once()


class ConsoleTests(unittest.TestCase):
    def test_cli_delegates_both_modes_to_agent(self) -> None:
        for meta_prompt in (False, True):
            with self.subTest(meta_prompt=meta_prompt):
                argv = [
                    "main.py", "--user", "Question", "--system", "System",
                    "--model", "custom-model", "--max-tokens", "100",
                    "--temperature", "0", "--stop-sequences", "END",
                    "--response-format", "object",
                ]
                if meta_prompt:
                    argv.append("--meta-prompt")
                output = io.StringIO()
                result = RequestResult(make_response('{"answer": "Final answer"}'), 0.5)
                with (
                    patch("sys.argv", argv),
                    patch("main.load_env") as load_env,
                    patch.dict("os.environ", {"API_KEY": "test-token"}),
                    patch("main.Agent") as agent_factory,
                    patch("main.console", Console(file=output, width=120)),
                ):
                    agent = agent_factory.return_value
                    agent.request.return_value = result
                    agent.request_with_meta_prompt.return_value = (
                        RequestResult(make_response("Optimized question"), 0.2), result,
                    )
                    main.main()

                    load_env.assert_called_once_with(main.ENV_PATH)
                    agent_factory.assert_called_once_with("test-token")
                    method = agent.request_with_meta_prompt if meta_prompt else agent.request
                    unused_method = agent.request if meta_prompt else agent.request_with_meta_prompt
                    method.assert_called_once_with(
                        user="Question", model="custom-model", system="System",
                        max_tokens=100, temperature=0.0, stop_sequences=["END"],
                        response_format="object",
                    )
                    unused_method.assert_not_called()
                self.assertIn("Final answer", output.getvalue())
                if meta_prompt:
                    self.assertIn("Optimized question", output.getvalue())


if __name__ == "__main__":
    unittest.main()
