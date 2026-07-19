"""Per-course fact-inventory cache shared by inspection and starter generation.

Cache files live beside the course index:
  data/indexes/{courseId}.facts.json

Invalidation:
- ``save_index`` / ``remove_index`` delete the cache (syllabus replacement)
- payload stores an ``indexFingerprint`` over chunk id+text+order; mismatch
  forces rebuild even if the file remains
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.storage import CourseArtifactStorage
from app.syllabus_facts import build_fact_inventory

FACT_INVENTORY_CACHE_VERSION = 1


def compute_index_fingerprint(raw_chunks: list[Any]) -> str:
    """Stable fingerprint of syllabus chunk evidence used for invalidation."""
    parts: list[str] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunkId") or chunk.get("id") or "").strip()
        text = str(chunk.get("text") or "")
        order = chunk.get("order", "")
        parts.append(f"{chunk_id}\n{order}\n{text}")
    digest = hashlib.sha256("\n--\n".join(parts).encode("utf-8")).hexdigest()
    return digest


def _inventory_payload_valid(
    payload: dict[str, Any],
    *,
    fingerprint: str,
) -> bool:
    if int(payload.get("cacheVersion") or 0) != FACT_INVENTORY_CACHE_VERSION:
        return False
    if str(payload.get("indexFingerprint") or "") != fingerprint:
        return False
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict):
        return False
    facts = inventory.get("facts")
    return isinstance(facts, list)


async def load_or_build_fact_inventory(
    *,
    course_id: str,
    raw_chunks: list[Any],
    storage: CourseArtifactStorage,
    force_refresh: bool = False,
    completion_fn=None,
    embed_fn=None,
    **build_kwargs: Any,
) -> dict[str, Any]:
    """Return fact inventory, reusing a valid per-course cache when possible.

    Returns a dict with:
    - all standard inventory fields (facts, factCount, ...)
    - ``cached``: True when served from disk without extraction
    - ``indexFingerprint``: fingerprint used for this inventory
    """
    fingerprint = compute_index_fingerprint(raw_chunks)

    if not force_refresh:
        cached_payload = storage.load_fact_inventory(course_id)
        if cached_payload is not None and _inventory_payload_valid(
            cached_payload, fingerprint=fingerprint
        ):
            inventory = dict(cached_payload["inventory"])
            inventory["cached"] = True
            inventory["indexFingerprint"] = fingerprint
            return inventory

    inventory = await build_fact_inventory(
        raw_chunks=raw_chunks,
        completion_fn=completion_fn,
        embed_fn=embed_fn,
        **build_kwargs,
    )
    # Persist only the inspectable inventory body (not call wrappers).
    persistable = {
        key: value
        for key, value in inventory.items()
        if key not in {"cached", "indexFingerprint"}
    }
    storage.save_fact_inventory(
        course_id,
        {
            "cacheVersion": FACT_INVENTORY_CACHE_VERSION,
            "indexFingerprint": fingerprint,
            "inventory": persistable,
        },
    )
    result = dict(persistable)
    result["cached"] = False
    result["indexFingerprint"] = fingerprint
    return result
