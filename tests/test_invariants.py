import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from llm_agent.invariants import (
    Invariant, InvariantSet, InvariantStorageError, InvariantStore, InvariantVerdict,
)


class InvariantStorageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "invariants.json"
        self.rules = InvariantSet((
            Invariant("stack", "Использовать Python"),
            Invariant("architecture", "Отделять ядро от UI"),
        ))

    def test_missing_file_does_not_create_configuration_and_saved_rules_survive_restart(self):
        self.assertEqual(InvariantStore(self.path).load(), InvariantSet())
        self.assertFalse(self.path.exists())
        InvariantStore(self.path).save(self.rules)
        self.assertEqual(InvariantStore(self.path).load(), self.rules)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), self.rules.to_dict())
        self.path.write_text(json.dumps(self.rules.to_dict()), encoding="utf-8-sig")
        self.assertEqual(InvariantStore(self.path).load(), self.rules)

    def test_rules_and_snapshots_cannot_be_mutated(self):
        with self.assertRaises(FrozenInstanceError):
            self.rules.rules[0].description = "Java"
        with self.assertRaises(FrozenInstanceError):
            self.rules.rules = ()
        exported = self.rules.to_dict()
        exported["rules"][0]["description"] = "Java"
        self.assertEqual(self.rules.rules[0].description, "Использовать Python")

    def test_invalid_configuration_never_becomes_empty_or_overwrites_file(self):
        rule = self.rules.to_dict()["rules"][0]
        malformed = [
            "not JSON", "[]", "null", '{"version":1,"rules":[],"rules":[]}',
            json.dumps({"version": True, "rules": []}),
            json.dumps({"version": 2, "rules": []}),
            json.dumps({"version": 1, "rules": {}, "extra": True}),
            json.dumps({"version": 1, "rules": [rule, rule]}),
            json.dumps({"version": 1, "rules": [{"id": "../x", "description": "test"}]}),
            json.dumps({"version": 1, "rules": [{"id": "x", "description": " "}]}),
            json.dumps({"version": 1, "rules": [{"id": "x", "description": []}]}),
            json.dumps({"version": 1, "rules": [dict(rule, enabled=False)]}),
        ]
        for content in malformed:
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                with self.assertRaises(InvariantStorageError):
                    InvariantStore(self.path).load()
                self.assertEqual(self.path.read_text(encoding="utf-8"), content)
        self.path.write_bytes(b"\xff\xff")
        with self.assertRaises(InvariantStorageError):
            InvariantStore(self.path).load()

    def test_write_failure_keeps_previous_rules_and_removes_temporary_file(self):
        store = InvariantStore(self.path)
        store.save(self.rules)
        original = self.path.read_bytes()
        with patch("pathlib.Path.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(InvariantStorageError):
                store.save(InvariantSet())
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_read_failure_is_not_treated_as_disabled_rules(self):
        with patch("pathlib.Path.read_text", side_effect=PermissionError("denied")):
            with self.assertRaises(InvariantStorageError):
                InvariantStore(self.path).load()


class InvariantVerdictTests(unittest.TestCase):
    def test_all_rules_must_be_checked_and_only_blocking_rules_are_shown(self):
        rules = InvariantSet((Invariant("stack", "Python"), Invariant("budget", "Бесплатные API")))
        verdict = InvariantVerdict.parse(json.dumps({"checks": [
            {"id": "stack", "status": "pass", "reason": "Совместимо"},
            {"id": "budget", "status": "conflict", "reason": "Секретный запрещённый рецепт"},
        ]}), rules)
        self.assertFalse(verdict.passed)
        refusal = verdict.refusal(rules, has_task=True)
        self.assertIn("[budget] Бесплатные API", refusal)
        self.assertNotIn("[stack]", refusal)
        self.assertNotIn("Секретный", refusal)
        self.assertIn("шаг сохранены без изменений", refusal)

    def test_duplicate_json_fields_cannot_override_conflict_with_pass(self):
        rules = InvariantSet((Invariant("stack", "Python"),))
        with self.assertRaises(ValueError):
            InvariantVerdict.parse(
                '{"checks":[{"id":"stack","status":"conflict","status":"pass","reason":"x"}]}',
                rules,
            )


if __name__ == "__main__":
    unittest.main()
