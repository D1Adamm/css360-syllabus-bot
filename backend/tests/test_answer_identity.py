"""What an answer's fingerprint covers: every input that decides the answer.

Ollama and the fine-tuned service are played by a dictionary of URL -> JSON;
the syllabus index is a real file in a temporary directory.
"""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi import HTTPException

from app import answer_identity as ai
from app.answer_identity import build_identity

COURSE = "css-360-winter-2026-a7rp"
MAIN = ai.OLLAMA_BASE_URL
FT_OLLAMA = "http://127.0.0.1:11435"
SERVICE = "http://127.0.0.1:9001"


def _index(text: str = "Homework is due Fridays.", model: str = "nomic-embed-text") -> dict[str, Any]:
    return {
        "embeddingModel": model,
        "createdAt": "2026-01-01T00:00:00Z",
        "chunks": [{"chunkId": "c1", "sectionTitle": "Homework", "text": text, "embedding": [0.1, 0.2]}],
    }


class _Storage:
    def __init__(self, root: Path) -> None:
        self.root = root

    def index_path(self, course_id: str) -> Path:
        return self.root / f"{course_id}.json"


class IdentityTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        ai.reset_identity_memo_for_tests()
        self.addCleanup(ai.reset_identity_memo_for_tests)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.write_index(_index())
        self.urls: dict[str, Any] = {
            f"{MAIN}/api/version": {"version": "0.33.2"},
            f"{MAIN}/api/tags": {"models": [
                {"name": "llama3.2:3b", "digest": "base-digest"},
                {"name": "nomic-embed-text:latest", "digest": "embed-digest"},
            ]},
            f"{SERVICE}/health": {
                "status": "ok",
                "ollamaUrl": FT_OLLAMA,
                "models": [{"courseId": COURSE, "version": "v2", "ollamaModel": "css360-v2", "available": True}],
            },
            f"{FT_OLLAMA}/api/version": {"version": "0.33.2"},
            f"{FT_OLLAMA}/api/tags": {"models": [{"name": "css360-v2:latest", "digest": "adapter-digest"}]},
        }
        self.requests: list[str] = []
        self.version: Any = {"version": "v2"}

        async def get_json(url: str) -> Any:
            self.requests.append(url)
            if url not in self.urls:
                raise ValueError("HTTP 503")
            return self.urls[url]

        def resolve(course_id: str) -> dict[str, Any]:
            if isinstance(self.version, Exception):
                raise self.version
            return self.version

        for target, new in (
            ("app.answer_identity._get_json", get_json),
            ("app.answer_identity.resolve_current_course_model", resolve),
            ("app.answer_identity.get_finetuned_service_url", lambda: SERVICE),
            ("app.answer_identity.get_course_artifact_storage", lambda: _Storage(self.root)),
        ):
            patcher = patch(target, new=new)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_index(self, data: dict[str, Any]) -> None:
        (self.root / f"{COURSE}.json").write_text(json.dumps(data), encoding="utf-8")

    async def fingerprint(self, condition: str, *, top_k: int = 4, fresh: bool = True) -> str:
        result = await build_identity(condition, COURSE, top_k=top_k, fresh=fresh)
        self.assertIsNotNone(result.identity, result.reason)
        return result.identity.fingerprint

    async def reason(self, condition: str) -> str | None:
        result = await build_identity(condition, COURSE, top_k=4, fresh=True)
        self.assertIsNone(result.identity)
        return result.reason


class FingerprintCoverageTests(IdentityTestCase):
    async def test_the_same_inputs_give_the_same_fingerprint(self) -> None:
        for condition in ai.CONDITIONS:
            with self.subTest(condition=condition):
                self.assertEqual(await self.fingerprint(condition), await self.fingerprint(condition))

    async def test_the_four_approaches_never_share_a_fingerprint(self) -> None:
        prints = {await self.fingerprint(condition) for condition in ai.CONDITIONS}
        self.assertEqual(len(prints), 4)

    async def _changes(self, conditions: tuple[str, ...], change) -> None:
        before = {c: await self.fingerprint(c) for c in ai.CONDITIONS}
        change()
        ai.reset_identity_memo_for_tests()
        for condition in ai.CONDITIONS:
            with self.subTest(condition=condition):
                after = await self.fingerprint(condition)
                if condition in conditions:
                    self.assertNotEqual(before[condition], after)
                else:
                    self.assertEqual(before[condition], after)

    async def test_a_rebuilt_base_model_retires_base_and_rag_only(self) -> None:
        def change() -> None:
            self.urls[f"{MAIN}/api/tags"]["models"][0]["digest"] = "base-digest-2"

        await self._changes(("base", "rag"), change)

    async def test_a_rebuilt_course_adapter_retires_the_fine_tuned_approaches_only(self) -> None:
        def change() -> None:
            self.urls[f"{FT_OLLAMA}/api/tags"]["models"][0]["digest"] = "adapter-digest-2"

        await self._changes(("fineTuned", "fineTunedRag"), change)

    async def test_a_new_embedding_model_retires_the_retrieval_approaches_only(self) -> None:
        def change() -> None:
            self.urls[f"{MAIN}/api/tags"]["models"][1]["digest"] = "embed-digest-2"

        await self._changes(("rag", "fineTunedRag"), change)

    async def test_an_ollama_upgrade_retires_what_that_server_answers(self) -> None:
        def main() -> None:
            self.urls[f"{MAIN}/api/version"] = {"version": "0.34.0"}

        # The embedding runs on the main server, so Fine-Tuned + RAG moves too.
        await self._changes(("base", "rag", "fineTunedRag"), main)

        def finetuned() -> None:
            self.urls[f"{FT_OLLAMA}/api/version"] = {"version": "0.34.0"}

        await self._changes(("fineTuned", "fineTunedRag"), finetuned)

    async def test_activating_another_course_version_retires_the_fine_tuned_approaches(self) -> None:
        def change() -> None:
            self.version = {"version": "v3"}
            self.urls[f"{SERVICE}/health"]["models"].append(
                {"courseId": COURSE, "version": "v3", "ollamaModel": "css360-v3", "available": True}
            )
            self.urls[f"{FT_OLLAMA}/api/tags"]["models"].append({"name": "css360-v3:latest", "digest": "v3-digest"})

        await self._changes(("fineTuned", "fineTunedRag"), change)

    async def test_a_replaced_syllabus_retires_the_retrieval_approaches_only(self) -> None:
        await self._changes(("rag", "fineTunedRag"), lambda: self.write_index(_index("Homework is due Mondays.")))

    async def test_new_embeddings_for_the_same_text_retire_the_retrieval_approaches(self) -> None:
        def change() -> None:
            data = _index()
            data["chunks"][0]["embedding"] = [0.3, 0.4]
            self.write_index(data)

        await self._changes(("rag", "fineTunedRag"), change)

    async def test_resaving_an_identical_index_retires_nothing(self) -> None:
        def change() -> None:
            data = _index()
            data["createdAt"] = "2026-06-01T00:00:00Z"
            self.write_index(data)

        await self._changes((), change)

    async def test_top_k_is_part_of_a_retrieval_answer_only(self) -> None:
        self.assertNotEqual(await self.fingerprint("rag", top_k=4), await self.fingerprint("rag", top_k=5))
        self.assertEqual(await self.fingerprint("base", top_k=4), await self.fingerprint("base", top_k=5))

    async def test_prompt_options_retrieval_code_and_host_are_each_covered(self) -> None:
        cases = (
            ("app.answer_identity.build_base_model_prompt", lambda q: f"NEW RULES {q}", ("base",)),
            ("app.answer_identity.build_grounded_prompt", lambda *a, **k: "NEW TEMPLATE", ("rag", "fineTunedRag")),
            ("app.answer_identity.classroom_options", lambda: {"temperature": 0, "num_predict": 64}, ("base", "rag")),
            ("app.answer_identity.CLASSROOM_NUM_PREDICT", 64, ("fineTuned", "fineTunedRag")),
            ("app.answer_identity.CLASSROOM_PROFILE_NAME", "classroom-v2", ai.CONDITIONS),
            ("app.answer_identity.RETRIEVAL_MODULES", ("course_rag.py",), ("rag", "fineTunedRag")),
            ("app.answer_identity.host_identity", lambda: "another cpu", ai.CONDITIONS),
            ("app.answer_identity.IDENTITY_VERSION", 2, ai.CONDITIONS),
        )
        before = {c: await self.fingerprint(c) for c in ai.CONDITIONS}
        for target, new, affected in cases:
            ai.reset_identity_memo_for_tests()
            with patch(target, new=new):
                for condition in ai.CONDITIONS:
                    with self.subTest(target=target, condition=condition):
                        changed = await self.fingerprint(condition) != before[condition]
                        self.assertEqual(changed, condition in affected)
        ai.reset_identity_memo_for_tests()

    def test_every_retrieval_module_the_grounded_path_imports_is_hashed(self) -> None:
        """A module added to retrieval must be added to RETRIEVAL_MODULES."""
        watched = {name[:-3] for name in ai.RETRIEVAL_MODULES}
        imported: set[str] = set()
        for name in ("course_rag.py", "grounded_rag.py"):
            tree = ast.parse((ai.APP_DIR / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app."):
                    imported.add(node.module.split(".", 1)[1])
        retrieval = {module for module in imported if module.startswith(("retrieval_", "rag", "course_rag"))}
        self.assertLessEqual(retrieval, watched)
        self.assertLessEqual({"course_rag", "rag", "retrieval_diversity", "retrieval_facets", "grounded_rag"}, watched)


class UnknownIdentityTests(IdentityTestCase):
    async def test_a_missing_base_model_means_no_identity(self) -> None:
        self.urls[f"{MAIN}/api/tags"]["models"].pop(0)
        self.assertEqual(await self.reason("base"), ai.REASON_BASE_MODEL)
        self.assertEqual(await self.reason("rag"), ai.REASON_BASE_MODEL)

    async def test_an_unreachable_ollama_means_no_identity(self) -> None:
        del self.urls[f"{MAIN}/api/version"]
        self.assertEqual(await self.reason("base"), ai.REASON_BASE_MODEL)

    async def test_a_missing_index_or_embedding_model(self) -> None:
        (self.root / f"{COURSE}.json").unlink()
        self.assertEqual(await self.reason("rag"), ai.REASON_NO_INDEX)
        self.assertEqual(await self.reason("fineTunedRag"), ai.REASON_NO_INDEX)
        self.write_index(_index())
        self.urls[f"{MAIN}/api/tags"]["models"].pop(1)
        ai.reset_identity_memo_for_tests()
        self.assertEqual(await self.reason("rag"), ai.REASON_EMBEDDING_MODEL)
        # Base needs no retrieval and is unaffected.
        await self.fingerprint("base")

    async def test_an_index_with_no_chunks_is_no_index(self) -> None:
        self.write_index({"embeddingModel": "nomic-embed-text", "chunks": []})
        self.assertEqual(await self.reason("rag"), ai.REASON_NO_INDEX)

    async def test_a_course_with_no_activated_model(self) -> None:
        self.version = HTTPException(status_code=409, detail="not activated")
        self.assertEqual(await self.reason("fineTuned"), ai.REASON_NO_COURSE_MODEL)
        self.assertEqual(await self.reason("fineTunedRag"), ai.REASON_NO_COURSE_MODEL)

    async def test_a_fine_tuned_service_that_is_down_unhealthy_or_too_old(self) -> None:
        health = self.urls.pop(f"{SERVICE}/health")
        self.assertEqual(await self.reason("fineTuned"), ai.REASON_FINETUNED_SERVICE)
        self.urls[f"{SERVICE}/health"] = {**health, "status": "unavailable"}
        self.assertEqual(await self.reason("fineTuned"), ai.REASON_FINETUNED_SERVICE)
        # A build that does not say which Ollama it uses cannot be verified.
        self.urls[f"{SERVICE}/health"] = {"status": "ok", "courses": []}
        self.assertEqual(await self.reason("fineTuned"), ai.REASON_FINETUNED_SERVICE)

    async def test_a_version_the_service_does_not_map_or_ollama_does_not_have(self) -> None:
        self.version = {"version": "v9"}
        self.assertEqual(await self.reason("fineTuned"), ai.REASON_FINETUNED_MODEL)
        self.version = {"version": "v2"}
        self.urls[f"{SERVICE}/health"]["models"][0]["available"] = False
        self.assertEqual(await self.reason("fineTuned"), ai.REASON_FINETUNED_MODEL)
        self.urls[f"{SERVICE}/health"]["models"][0]["available"] = True
        self.urls[f"{FT_OLLAMA}/api/tags"] = {"models": []}
        self.assertEqual(await self.reason("fineTuned"), ai.REASON_FINETUNED_MODEL)

    async def test_no_fine_tuned_service_configured(self) -> None:
        with patch("app.answer_identity.get_finetuned_service_url", new=lambda: None):
            self.assertEqual(await self.reason("fineTuned"), ai.REASON_FINETUNED_SERVICE)


class CostTests(IdentityTestCase):
    async def test_repeated_lookups_make_no_http_calls_until_asked_fresh(self) -> None:
        for condition in ai.CONDITIONS:
            await build_identity(condition, COURSE, top_k=4)
        first = len(self.requests)
        self.assertGreater(first, 0)
        for _ in range(5):
            for condition in ai.CONDITIONS:
                await build_identity(condition, COURSE, top_k=4)
        self.assertEqual(len(self.requests), first)
        await build_identity("base", COURSE, top_k=4, fresh=True)
        self.assertGreater(len(self.requests), first)

    async def test_the_fine_tuned_digest_comes_from_the_ollama_the_service_names(self) -> None:
        result = await build_identity("fineTuned", COURSE, top_k=4)
        self.assertEqual(result.identity.fields["model"], {"tag": "css360-v2:latest", "digest": "adapter-digest", "courseVersion": "v2"})
        self.assertEqual((result.identity.model_version, result.identity.model_tag), ("v2", "css360-v2:latest"))
        self.assertIn(f"{FT_OLLAMA}/api/tags", self.requests)

    async def test_the_index_is_rehashed_only_when_its_file_changes(self) -> None:
        with patch("app.answer_identity._hash_index", wraps=ai._hash_index) as hashed:
            for _ in range(4):
                ai.index_fingerprint(COURSE)
            self.assertEqual(hashed.call_count, 1)
            self.write_index(_index("Changed text, different size."))
            ai.index_fingerprint(COURSE)
            self.assertEqual(hashed.call_count, 2)

    def test_tags_are_compared_as_ollama_names_them(self) -> None:
        self.assertEqual(ai.ollama_tag("nomic-embed-text"), "nomic-embed-text:latest")
        self.assertEqual(ai.ollama_tag("llama3.2:3b"), "llama3.2:3b")
        self.assertEqual(ai.ollama_tag("registry.local:5000/css360-v2"), "registry.local:5000/css360-v2:latest")


if __name__ == "__main__":
    unittest.main()
