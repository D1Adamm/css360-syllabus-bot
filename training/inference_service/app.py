"""Per-course fine-tuned (QLoRA) inference service.

Loads the base model once and attaches one LoRA adapter per course on top of it,
choosing between them per request. Adapters are never merged into the base.

Why one process and many adapters
---------------------------------
Three shapes were possible: one process per course, one process that reloads a
single adapter on demand, or one process holding several adapters at once. The
last is both the simplest and the only one that is cheap, and the repository
already had everything needed for it.

A 4-bit Llama-3.2-3B base is ~2.5 GB on the GPU. A LoRA adapter for it is ~47 MB
— the CSS 350 adapter is exactly that. PEFT keeps adapters in a named registry
on the loaded model and switches between them with `set_adapter`, so serving a
second course costs a rounding error of GPU memory and no second allocation.
One process per course would cost a whole GPU per course on a shared cluster;
reloading one adapter per request would put a multi-second load in front of
every question.

Course isolation, concretely
----------------------------
1. Every request names its course. There is no default and no fallback: a
   request without a course id is a 422, not a request served by whichever
   adapter happened to be current.
2. The adapter is resolved from the serving root by course id and version, and
   the path is built from validated components — there is no request that
   reaches another course's directory.
3. `set_adapter` and `generate` happen together under one lock. FastAPI runs
   sync handlers in a thread pool, so without it two concurrent requests could
   interleave "select CSS 350" and "generate", and CSS 350's question would be
   answered by CSS 360's adapter. That is the exact failure this whole design
   exists to prevent, and it would be invisible in the response.
4. The response echoes the course id and version it actually used, and the
   backend refuses a response whose course does not match what it asked for.

Classroom GPU mode
------------------
The same process also answers the Base and RAG conditions (`target: "base"`):
the base model with every adapter switched off. Every answer, either target,
is decoded exactly as the VM decodes it: the prompt rendered as Ollama renders
llama3.2 (`helpers.render_production_prompt`), greedy, the request's output cap
(the classroom profile's 128), Ollama's stop tokens, a 4096-token context, a
1.05 repetition penalty over the whole context, seed 360. Each response echoes
those settings (`decoding`) and the backend refuses one that differs from what
it would have sent the VM. Weights are the same checkpoint in a different
quantisation (bitsandbytes NF4 here, GGUF Q4_K_M on the VM), so answers can
differ in wording; every setting that is a choice matches.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from helpers import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_CTX,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SEED,
    BOS_TOKEN,
    END_OF_GENERATION_TOKENS,
    STOP_TOKENS,
    CourseAdapterError,
    decoding_summary,
    render_production_prompt,
    resolve_max_new_tokens,
    with_single_bos,
    assert_hf_auth_available,
    list_available_courses,
    resolve_course_adapter,
    resolve_max_loaded_adapters,
    resolve_model_id,
    resolve_port,
    resolve_serving_root,
    resolve_session_deadline,
    validate_question,
)


class GenerateRequest(BaseModel):
    question: str = Field(..., description="Student question to answer")
    course_id: Optional[str] = Field(
        default=None,
        alias="courseId",
        description=(
            "Which course's adapter to answer with. Required for the course "
            "target: there is no course-agnostic fine-tuned model."
        ),
    )
    target: Literal["course", "base"] = Field(
        default="course",
        description=(
            "`course`: the course's adapter (Fine-Tuned, Fine-Tuned + RAG). "
            "`base`: the base model with every adapter off (Base, RAG)."
        ),
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        alias="maxNewTokens",
        ge=1,
        le=DEFAULT_MAX_NEW_TOKENS,
        description="Output cap for this answer (the classroom profile sends 128).",
    )
    model_version: Optional[str] = Field(
        default=None,
        alias="modelVersion",
        description=(
            "The registered version to use, e.g. v1. Omit to use whatever the "
            "course was last published with."
        ),
    )

    model_config = {"populate_by_name": True}


class GenerateResponse(BaseModel):
    answer: str
    model: str
    courseId: Optional[str] = None
    modelVersion: Optional[str] = None
    adapterLoaded: bool
    generationSeconds: float
    target: str = "course"
    engine: str = "tillicum-transformers"
    #: The settings this answer was decoded with (Ollama's vocabulary).
    decoding: Dict[str, Any]
    #: Token counts and milliseconds, under the backend's log names.
    timings: Dict[str, int]


class CourseSummary(BaseModel):
    courseId: str
    versions: List[str]
    currentVersion: str


class HealthResponse(BaseModel):
    status: str
    model: str
    servingRoot: str
    #: True when the base model is up. Named `adapterLoaded` because every
    #: existing health check — the start script, the tunnel script, the backend
    #: — reads that key, and a rename would silently break all three.
    adapterLoaded: bool
    cudaAvailable: bool
    hostname: str
    port: int
    courses: List[CourseSummary]
    loadedAdapters: List[str]
    expiresAt: Optional[float] = None
    secondsRemaining: Optional[float] = None
    #: What classroom GPU mode checks before switching to this service.
    engine: str = "tillicum-transformers"
    targets: List[str] = ["course", "base"]
    decoding: Dict[str, Any] = {}
    gpuName: Optional[str] = None
    #: Answers served since start, per target. Lets a switch prove that the
    #: backend's requests actually reached this GPU.
    served: Dict[str, int] = {}


def assert_cuda_available() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The fine-tuned inference service requires one GPU."
        )


def _model_device(model: Any) -> Any:
    try:
        return model.device
    except Exception:  # noqa: BLE001
        return next(model.parameters()).device


class InferenceEngine:
    """One base model, a bounded set of course adapters loaded on top of it."""

    def __init__(
        self,
        *,
        model_id: str,
        serving_root: Path,
        max_loaded_adapters: int,
    ) -> None:
        self.model_id = model_id
        self.serving_root = serving_root
        self.max_loaded_adapters = max_loaded_adapters
        self.tokenizer: Any = None
        self.model: Any = None
        self.base_loaded = False
        #: adapterKey -> adapter directory, in least-recently-used order.
        self._adapters: "OrderedDict[str, str]" = OrderedDict()
        #: Guards adapter selection *and* generation together. Splitting them
        #: would reintroduce the cross-course bug this lock exists to prevent.
        self._lock = threading.RLock()
        self.served: Dict[str, int] = {"course": 0, "base": 0}

    # -- lifecycle -------------------------------------------------------- #

    def load_base(self) -> None:
        """Load the base model once. No adapter is attached here.

        Deliberately separate from adapter loading: the service is up and
        answering `/health` as soon as the base is ready, and a course whose
        adapter has not been published yet is a per-request 409 rather than a
        service that refuses to start.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        assert_cuda_available()
        assert_hf_auth_available()

        use_bf16 = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        print(f"Loading tokenizer/model: {self.model_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
        )

        self.tokenizer = tokenizer
        self.model = model
        self.base_loaded = True
        print(f"Base model ready. Serving root: {self.serving_root}", flush=True)

    def loaded_adapter_keys(self) -> List[str]:
        with self._lock:
            return list(self._adapters)

    # -- adapters --------------------------------------------------------- #

    def _attach(self, key: str, path: Path) -> None:
        """Load one adapter onto the base model under `key`.

        The first adapter turns the base into a `PeftModel`; every subsequent
        one is added to that same model. Both paths end with the adapter
        registered under a name that includes its course *and* its version, so a
        promotion is a different key rather than a silent overwrite.
        """
        from peft import PeftModel

        if isinstance(self.model, PeftModel):
            self.model.load_adapter(str(path), adapter_name=key)
        else:
            self.model = PeftModel.from_pretrained(
                self.model, str(path), adapter_name=key
            )
        self.model.eval()
        self._adapters[key] = str(path)
        print(f"Loaded adapter {key} from {path}", flush=True)

    def _evict_if_needed(self) -> None:
        """Drop least-recently-used adapters down to the bound.

        Never evicts the adapter that is about to be used: this runs before the
        new one is attached, and the caller has already moved any hit to the
        most-recent end.
        """
        while len(self._adapters) > self.max_loaded_adapters:
            key, _path = self._adapters.popitem(last=False)
            try:
                self.model.delete_adapter(key)
                print(f"Evicted adapter {key}", flush=True)
            except Exception as exc:  # noqa: BLE001
                # An adapter that will not unload is a leak, not a wrong answer:
                # it stays resident but is no longer selectable through the
                # cache. Reported rather than raised so one bad eviction cannot
                # fail an unrelated request.
                print(f"WARNING: could not unload adapter {key}: {exc}", flush=True)

    def ensure_adapter(self, course_id: str, version: Optional[str]) -> Dict[str, Any]:
        """Resolve, load if needed, and return which adapter answers for a course."""
        resolved = resolve_course_adapter(course_id, version, root=self.serving_root)
        key = resolved["adapterKey"]

        with self._lock:
            if key in self._adapters:
                self._adapters.move_to_end(key)
                return resolved
            self._attach(key, resolved["path"])
            self._adapters.move_to_end(key)
            self._evict_if_needed()
            return resolved

    # -- generation ------------------------------------------------------- #

    def stop_token_ids(self) -> List[int]:
        """Ollama's llama3.2 stop tokens plus the tokenizer's end of sequence."""
        ids: List[int] = []
        for token in (*STOP_TOKENS, *END_OF_GENERATION_TOKENS):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
                ids.append(token_id)
        eos = self.tokenizer.eos_token_id
        for token_id in eos if isinstance(eos, list) else [eos]:
            if isinstance(token_id, int) and token_id not in ids:
                ids.append(token_id)
        return ids

    def generate(
        self,
        question: str,
        *,
        target: str = "course",
        course_id: Optional[str] = None,
        model_version: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: int = DEFAULT_SEED,
    ) -> Dict[str, Any]:
        """One answer, decoded the way the VM decodes it.

        `target="course"` answers with the course's adapter; `target="base"`
        with the base model and every adapter switched off. The prompt is the
        production rendering, not `apply_chat_template` (see helpers).
        """
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")
        if target not in ("course", "base"):
            raise ValueError(f"Unknown target: {target!r}")

        cleaned = validate_question(question)
        cap = resolve_max_new_tokens(max_new_tokens)
        resolved: Optional[Dict[str, Any]] = None
        if target == "course":
            if not course_id:
                raise CourseAdapterError("courseId is required for the course target.")
            resolved = self.ensure_adapter(course_id, model_version)

        prompt = render_production_prompt(cleaned)
        bos_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        token_ids = with_single_bos(
            self.tokenizer(prompt, add_special_tokens=False)["input_ids"], bos_id
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        # num_ctx, as on the VM: the prompt and the answer share the window.
        # A prompt too long for it keeps its first token (BOS) and its end.
        limit = DEFAULT_NUM_CTX - 1
        if input_ids.shape[-1] > limit:
            input_ids = torch.cat([input_ids[:, :1], input_ids[:, -(limit - 1):]], dim=-1)
            print(f"WARNING: prompt truncated to {limit} tokens", flush=True)
        prompt_length = int(input_ids.shape[-1])
        cap = min(cap, DEFAULT_NUM_CTX - prompt_length)

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": cap,
            "do_sample": False,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": self.stop_token_ids(),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "attention_mask": torch.ones_like(input_ids),
        }

        # Selection and generation are one critical section. FastAPI runs sync
        # handlers concurrently in a thread pool, so a second request selecting
        # its own adapter between these two statements would answer this one
        # with the wrong course's weights — or, for a base request, with any.
        with self._lock:
            device = _model_device(self.model)
            located = input_ids.to(device)
            generate_kwargs["attention_mask"] = generate_kwargs["attention_mask"].to(device)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            start = time.perf_counter()
            with torch.inference_mode():
                if target == "course":
                    self.model.set_adapter(resolved["adapterKey"])
                    output_ids = self.model.generate(input_ids=located, **generate_kwargs)
                elif hasattr(self.model, "disable_adapter"):
                    with self.model.disable_adapter():
                        output_ids = self.model.generate(input_ids=located, **generate_kwargs)
                else:
                    output_ids = self.model.generate(input_ids=located, **generate_kwargs)
            elapsed = time.perf_counter() - start
            self.served[target] += 1

        completion_ids = output_ids[0, prompt_length:]
        answer = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return {
            "answer": answer,
            "target": target,
            "courseId": resolved["courseId"] if resolved else course_id,
            "modelVersion": resolved["version"] if resolved else None,
            "generationSeconds": elapsed,
            "decoding": decoding_summary(cap),
            "timings": {
                "prompt_tokens": prompt_length,
                "output_tokens": int(completion_ids.shape[-1]),
                "eval_ms": int(elapsed * 1000),
                "ollama_total_ms": int(elapsed * 1000),
            },
        }


ENGINE = InferenceEngine(
    model_id=resolve_model_id(),
    serving_root=resolve_serving_root(),
    max_loaded_adapters=resolve_max_loaded_adapters(),
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import socket

    hostname = socket.gethostname()
    port = resolve_port()
    print(f"Starting inference service on host={hostname} port={port}", flush=True)
    try:
        ENGINE.load_base()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR during model load: {exc}", flush=True)
        raise
    available = list_available_courses(root=ENGINE.serving_root)
    print(
        "Courses with a published adapter: {0}".format(
            ", ".join(item["courseId"] for item in available) or "none"
        ),
        flush=True,
    )
    print(f"Listening on http://0.0.0.0:{port} (hostname={hostname})", flush=True)
    yield


app = FastAPI(
    title="Per-Course Fine-Tuned Inference Service",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    import socket

    import torch

    deadline = resolve_session_deadline()
    return HealthResponse(
        status="ok" if ENGINE.base_loaded else "loading",
        model=ENGINE.model_id,
        servingRoot=str(ENGINE.serving_root),
        adapterLoaded=ENGINE.base_loaded,
        cudaAvailable=torch.cuda.is_available(),
        hostname=socket.gethostname(),
        port=resolve_port(),
        courses=[
            CourseSummary(**item) for item in list_available_courses(root=ENGINE.serving_root)
        ],
        loadedAdapters=ENGINE.loaded_adapter_keys(),
        expiresAt=deadline,
        secondsRemaining=(deadline - time.time()) if deadline is not None else None,
        decoding=decoding_summary(DEFAULT_MAX_NEW_TOKENS),
        gpuName=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        served=dict(ENGINE.served),
    )


@app.get("/courses")
def courses() -> Dict[str, Any]:
    """Which courses this service can answer for, and with which versions."""
    return {
        "servingRoot": str(ENGINE.serving_root),
        "courses": list_available_courses(root=ENGINE.serving_root),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest) -> GenerateResponse:
    try:
        question = validate_question(body.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if body.target == "course" and not (body.course_id or "").strip():
        raise HTTPException(
            status_code=422,
            detail="courseId is required: there is no course-agnostic fine-tuned model.",
        )

    if not ENGINE.base_loaded or ENGINE.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        result = ENGINE.generate(
            question,
            target=body.target,
            course_id=body.course_id,
            model_version=body.model_version,
            max_new_tokens=body.max_new_tokens,
        )
    except CourseAdapterError as exc:
        # 409 rather than 500: the service is healthy and the request is well
        # formed. What is missing is a published adapter for this course, which
        # is a state an operator fixes with the promote script.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    print(
        "generation target={0} course={1} version={2} prompt_tokens={3} "
        "output_tokens={4} elapsed_ms={5}".format(
            result["target"],
            result["courseId"] or "-",
            result["modelVersion"] or "-",
            result["timings"]["prompt_tokens"],
            result["timings"]["output_tokens"],
            result["timings"]["eval_ms"],
        ),
        flush=True,
    )
    return GenerateResponse(
        answer=result["answer"],
        model=ENGINE.model_id,
        courseId=result["courseId"],
        modelVersion=result["modelVersion"],
        adapterLoaded=True,
        generationSeconds=result["generationSeconds"],
        target=result["target"],
        decoding=result["decoding"],
        timings=result["timings"],
    )


def main() -> None:
    import socket

    import uvicorn

    host = (os.environ.get("INFERENCE_HOST") or "0.0.0.0").strip()
    port = resolve_port()
    hostname = socket.gethostname()
    print(f"Node hostname: {hostname}", flush=True)
    print(f"Listening port: {port}", flush=True)
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
