"""The one Llama 3 prompt rendering that training and serving share.

Production answers every condition through Ollama's `/api/chat` with one user
message, which Ollama renders with the llama3.2 model's own template:

    <|start_header_id|>system<|end_header_id|>\\n\\nCutting Knowledge Date: December 2023\\n\\n<|eot_id|>
    <|start_header_id|>user<|end_header_id|>\\n\\n{content}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\\n\\n

with exactly one `<|begin_of_text|>` added by the tokenizer. The Tillicum GPU
service has reproduced this string since the classroom work (checked against
Ollama 0.33.2: `/api/generate` with `raw: true` and this text gives the same
answer and the same prompt token count as `/api/chat`).

Training did not. `train_qlora.py` rendered its examples with the Transformers
tokenizer's `apply_chat_template`, which adds a `Today Date: <today>` line to
the system block and puts `<|begin_of_text|>` into the text itself. So every
adapter so far was trained on a system block the model never sees in
production, one that changed from day to day, and with the BOS token possibly
twice (audit finding B14, 2026-11-03). This module is now the only definition:
the GPU service imports it, the trainer imports it, and a test on each side
pins the rendered string.

Pure Python, no third-party imports, so a Slurm job, the VM's service and a
unit test all read the same file.
"""

from __future__ import annotations

from typing import List, Tuple

#: Named in every runtime report and decoding summary so an adapter and the
#: service that runs it can be checked for the same format.
PROMPT_FORMAT = "ollama-llama3.2-chat"

BOS_TOKEN = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
#: Closes every turn, and is what the model emits to end an answer.
END_OF_TURN = "<|eot_id|>"

SYSTEM_BLOCK = (
    START_HEADER + "system" + END_HEADER + "\n\n"
    "Cutting Knowledge Date: December 2023\n\n"
    + END_OF_TURN
)

#: Where the answer starts. Training's completion-only loss masks everything
#: up to and including this; serving stops generation at any of STOP_TOKENS.
ASSISTANT_HEADER = START_HEADER + "assistant" + END_HEADER + "\n\n"

#: Ollama's llama3.2 stop parameters. Generation also ends at the tokenizer's
#: end-of-sequence tokens.
STOP_TOKENS = (START_HEADER, END_HEADER, END_OF_TURN)
#: Tokens llama.cpp treats as end of generation for Llama 3 on its own,
#: whatever the stop parameters say.
END_OF_GENERATION_TOKENS = (END_OF_TURN, "<|eom_id|>", "<|end_of_text|>")

#: Every special token the rendering relies on. A tokenizer that does not map
#: each of these to exactly one id would train (or serve) a different sequence.
REQUIRED_SPECIAL_TOKENS = (BOS_TOKEN, START_HEADER, END_HEADER, END_OF_TURN)


def render_production_prompt(content: str) -> str:
    """One user turn, rendered exactly as Ollama renders it for llama3.2.

    No `<|begin_of_text|>`: the tokenizer adds it (`with_single_bos` on the GPU
    service; llama.cpp on the VM; the Transformers tokenizer in training).
    """
    return (
        SYSTEM_BLOCK
        + START_HEADER + "user" + END_HEADER + "\n\n"
        + content
        + END_OF_TURN
        + ASSISTANT_HEADER
    )


def render_training_completion(response: str) -> str:
    """The assistant turn as training targets it: the answer, then end-of-turn."""
    return response + END_OF_TURN


def render_training_halves(instruction: str, response: str) -> Tuple[str, str]:
    """`(prompt, completion)`: the production prompt and the closed answer.

    Concatenated they are the whole training text; TRL trains on the
    completion alone when given the two halves.
    """
    return render_production_prompt(instruction), render_training_completion(response)


def render_training_text(instruction: str, response: str) -> str:
    """The whole text one training example becomes, BOS excluded."""
    prompt, completion = render_training_halves(instruction, response)
    return prompt + completion


def with_single_bos(token_ids: List[int], bos_id: int) -> List[int]:
    """Exactly one leading BOS, as llama.cpp tokenises the prompt on the VM.

    Whether a Transformers tokenizer adds BOS depends on how it was built (the
    Hub tokenizer's post-processor does; one rebuilt from a GGUF did not), so
    neither the service nor the trainer relies on it either way.
    """
    ids = list(token_ids)
    while ids and ids[0] == bos_id:
        ids.pop(0)
    return [bos_id] + ids
