from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ollama import generate_starter_ollama_completion  # noqa: E402
from app.syllabus_facts import (  # noqa: E402
    SEED_GENERATION_MODEL,
    batch_section_groups,
    build_fact_extraction_prompt,
    build_section_groups,
    _parse_facts_payload,
)

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "indexes"


async def main() -> None:
    data = json.loads((INDEX_DIR / "css-360-winter-2026-a7rp.json").read_text())
    chunks = data.get("chunks", [])
    groups = build_section_groups(chunks)
    batches = batch_section_groups(groups, char_budget=6000)
    print(f"num_batches={len(batches)}", flush=True)

    batch = batches[0]
    prompt = build_fact_extraction_prompt(batch)
    print(f"prompt_chars={len(prompt)}", flush=True)
    gen = await generate_starter_ollama_completion(
        prompt, model=SEED_GENERATION_MODEL, response_format="json", think=False
    )
    answer = gen.get("answer", "")
    print(f"answer_len={len(answer)}", flush=True)
    print("=== RAW ANSWER (first 2000) ===", flush=True)
    print(answer[:2000], flush=True)
    try:
        facts = _parse_facts_payload(answer)
        print(f"parsed_facts={len(facts)}", flush=True)
        if facts:
            print(json.dumps(facts[0], indent=2), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"PARSE_ERROR: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
