"""Контракт NVD MCP: реальные преобразования, внешний HTTP заменён mock."""

import asyncio
import logging
import threading
import unittest
from unittest.mock import patch

import requests
from mcp.server.mcpserver.exceptions import ToolError

from VulnSearchMCPServerExample import nvd_api, server


def cve(number=12345, **overrides):
    return {
        "id": f"CVE-2024-{number}",
        "descriptions": [
            {"lang": "es", "value": "Descripción"},
            {"lang": "en", "value": "PyTorch vulnerability"},
        ],
        "published": "2024-01-01T00:00:00.000",
        "lastModified": "2024-02-01T00:00:00.000",
        "vulnStatus": "Analyzed",
        "metrics": {"cvssMetricV31": [{
            "type": "Primary",
            "cvssData": {"version": "3.1", "baseScore": 8.8, "baseSeverity": "HIGH"},
        }]},
        **overrides,
    }


class NISTServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"NIST_API_TOKEN": "test-private-nist-token"})
        self.env.start()
        self.addCleanup(self.env.stop)
        # Реальные блокировки и запросы nvdlib сохраняются; пропускаем только паузу квоты.
        self.sleep = patch.object(nvd_api.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    async def test_success_returns_cve_data_and_limited_api_call(self):
        with patch.object(nvd_api.nvdlib, "searchCVE", return_value=[cve()]) as api:
            result = await server.search_vulnerabilities("  PyTorch  ", 5)
        api.assert_called_once_with(
            keywordSearch="PyTorch", noRejected=True, limit=5,
            key="test-private-nist-token", delay=0.7, asDict=True,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["project_name"], "PyTorch")
        self.assertEqual(result["returned_count"], 1)
        self.assertFalse(result["limit_reached"])
        self.assertIn("не доказывает", result["search_note"])
        self.assertEqual(result["vulnerabilities"], [{
            "id": "CVE-2024-12345", "description": "PyTorch vulnerability",
            "url": "https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
            "cvss": {"version": "3.1", "score": 8.8, "severity": "HIGH"},
            "published": "2024-01-01T00:00:00.000",
            "last_modified": "2024-02-01T00:00:00.000", "status": "Analyzed",
        }])

    async def test_empty_success_is_explicit_no_results(self):
        with patch.object(nvd_api.nvdlib, "searchCVE", return_value=[]):
            result = await server.search_vulnerabilities("zero-cves")
        self.assertEqual(result["status"], "no_results")
        self.assertEqual(result["returned_count"], 0)
        self.assertEqual(result["vulnerabilities"], [])

    async def test_limit_is_enforced_even_if_library_returns_extra_records(self):
        records = [cve(10000 + index) for index in range(12)]
        with patch.object(nvd_api.nvdlib, "searchCVE", return_value=records) as api:
            result = await server.search_vulnerabilities("PyTorch", 10)
        self.assertEqual(api.call_args.kwargs["limit"], 10)
        self.assertEqual(result["returned_count"], 10)
        self.assertTrue(result["limit_reached"])
        self.assertEqual(result["vulnerabilities"][-1]["id"], "CVE-2024-10009")

    async def test_keyword_is_encoded_as_one_parameter(self):
        with patch.object(nvd_api.nvdlib, "searchCVE", return_value=[]) as api:
            result = await server.search_vulnerabilities(" Product   & key=secret ")
        self.assertEqual(result["query"], "Product & key=secret")
        self.assertEqual(api.call_args.kwargs["keywordSearch"], "Product%20%26%20key%3Dsecret")

    async def test_missing_metrics_are_null_and_description_has_language_fallback(self):
        record = cve(metrics={}, descriptions=[{"lang": "es", "value": "Descripción"}])
        with patch.object(nvd_api.nvdlib, "searchCVE", return_value=[record]):
            result = await server.search_vulnerabilities("PyTorch")
        item = result["vulnerabilities"][0]
        self.assertEqual(item["description"], "Descripción")
        self.assertEqual(item["cvss"], {"version": None, "score": None, "severity": None})

    async def test_cvss_prefers_newest_version_and_primary_entry(self):
        record = cve()
        record["metrics"]["cvssMetricV40"] = [
            {"type": "Secondary", "cvssData": {"baseScore": 6.0, "baseSeverity": "MEDIUM"}},
            {"type": "Primary", "cvssData": {"baseScore": 9.3, "baseSeverity": "CRITICAL"}},
        ]
        with patch.object(nvd_api.nvdlib, "searchCVE", return_value=[record]):
            result = await server.search_vulnerabilities("PyTorch")
        self.assertEqual(result["vulnerabilities"][0]["cvss"], {
            "version": "4.0", "score": 9.3, "severity": "CRITICAL",
        })

    async def test_cvss_v2_severity_is_read_outside_cvss_data(self):
        record = cve(metrics={"cvssMetricV2": [{
            "baseSeverity": "HIGH", "cvssData": {"baseScore": 7.5},
        }]})
        with patch.object(nvd_api.nvdlib, "searchCVE", return_value=[record]):
            result = await server.search_vulnerabilities("PyTorch")
        self.assertEqual(result["vulnerabilities"][0]["cvss"], {
            "version": "2.0", "score": 7.5, "severity": "HIGH",
        })

    async def test_invalid_arguments_never_call_api(self):
        with patch.object(nvd_api.nvdlib, "searchCVE") as api:
            for name, limit in [("", 5), ("  ", 5), ("x" * 201, 5), ("Py\nTorch", 5),
                                ("PyTorch", 4), ("PyTorch", 11), ("PyTorch", 5.0), ("PyTorch", True)]:
                with self.subTest(name=name, limit=limit), self.assertRaises(ToolError):
                    await server.search_vulnerabilities(name, limit)
        api.assert_not_called()

    async def test_missing_key_is_configuration_error(self):
        with patch.dict("os.environ", {"NIST_API_TOKEN": ""}), \
                patch.object(nvd_api.nvdlib, "searchCVE") as api:
            with self.assertRaisesRegex(ToolError, "NIST_API_TOKEN"):
                await server.search_vulnerabilities("PyTorch")
        api.assert_not_called()

    async def test_api_errors_are_redacted_and_never_no_results(self):
        for status in [401, 403, 429, 500]:
            response = requests.Response()
            response.status_code = status
            error = requests.HTTPError("test-private-nist-token private HTTP body", response=response)
            with self.subTest(status=status), \
                    patch.object(nvd_api.nvdlib, "searchCVE", side_effect=error) as api:
                with self.assertRaises(ToolError) as raised:
                    await server.search_vulnerabilities("PyTorch")
                self.assertNotIn("test-private-nist-token", str(raised.exception))
                self.assertNotIn("private HTTP body", str(raised.exception))
                self.assertIn("неизвестен", str(raised.exception))
                api.assert_called_once()

    async def test_connection_and_socket_timeout_errors_are_redacted(self):
        for error in [requests.Timeout("test-private-nist-token"),
                      requests.ConnectionError("test-private-nist-token"),
                      LookupError("test-private-nist-token")]:
            with self.subTest(error=type(error).__name__), \
                    patch.object(nvd_api.nvdlib, "searchCVE", side_effect=error):
                with self.assertRaises(ToolError) as raised:
                    await server.search_vulnerabilities("PyTorch")
                self.assertNotIn("test-private-nist-token", str(raised.exception))
                self.assertIn("неизвестен", str(raised.exception))

    async def test_nvdlib_swallowed_invalid_json_is_not_no_results(self):
        response = requests.Response()
        response.status_code = 200
        response._content = b"<html>unavailable</html>"
        response.url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        log = logging.getLogger("nvdlib.get")
        old_handlers = list(log.handlers)
        old_level = log.level
        with patch("nvdlib.get.requests.get", return_value=response) as get:
            with self.assertRaisesRegex(ToolError, "некорректный ответ"):
                await server.search_vulnerabilities("PyTorch")
        get.assert_called_once()
        self.assertEqual(get.call_args.kwargs["timeout"], 30)
        self.assertEqual(log.handlers, old_handlers)
        self.assertEqual(log.level, old_level)

    async def test_real_nvdlib_successful_empty_response_is_no_results(self):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"totalResults":0,"resultsPerPage":0,"vulnerabilities":[]}'
        with patch("nvdlib.get.requests.get", return_value=response):
            result = await server.search_vulnerabilities("zero-cves")
        self.assertEqual(result["status"], "no_results")

    async def test_invalid_cve_payload_is_error_not_partial_success(self):
        for records in [None, [{"id": "wrong"}], [cve(), {"id": "CVE-2024-10000", "metrics": None}]]:
            with self.subTest(records=records), \
                    patch.object(nvd_api.nvdlib, "searchCVE", return_value=records):
                with self.assertRaisesRegex(ToolError, "некорректный ответ"):
                    await server.search_vulnerabilities("PyTorch")

    async def test_sync_search_runs_off_event_loop(self):
        event_loop_thread = threading.get_ident()
        worker_threads = []

        def search(**kwargs):
            worker_threads.append(threading.get_ident())
            return []

        with patch.object(nvd_api.nvdlib, "searchCVE", side_effect=search):
            await server.search_vulnerabilities("PyTorch")
        self.assertNotEqual(worker_threads, [event_loop_thread])

    async def test_timeout_keeps_worker_lock_until_request_finishes(self):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def slow_search(**kwargs):
            entered.set()
            try:
                release.wait(timeout=2)
                return []
            finally:
                finished.set()

        with patch.object(nvd_api.nvdlib, "searchCVE", side_effect=slow_search) as api, \
                patch.object(server, "SEARCH_TIMEOUT_SECONDS", 0.05):
            try:
                with self.assertRaisesRegex(ToolError, "время поиска"):
                    await server.search_vulnerabilities("PyTorch")
                self.assertTrue(entered.is_set())
                with self.assertRaisesRegex(ToolError, "занят"):
                    await server.search_vulnerabilities("Visual Studio Code")
                api.assert_called_once()
            finally:
                release.set()
                await asyncio.to_thread(finished.wait, 1)
        # Executor завершится в tearDown IsolatedAsyncioTestCase; дождёмся также
        # finally функции search, чтобы следующий тест не унаследовал блокировку.
        await asyncio.to_thread(nvd_api._request_lock.acquire)
        nvd_api._request_lock.release()


if __name__ == "__main__":
    unittest.main()
