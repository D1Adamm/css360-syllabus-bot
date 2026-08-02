import os

# Load backend/.env before other app modules read configuration.
from app import config as _config  # noqa: F401
from app.config import load_backend_env

load_backend_env()

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.course_id import assert_valid_course_id
from app.course_index import build_course_rag_index
from app.course_rag import generate_course_rag_answer
from app.finetuned_client import (
    check_finetuned_service_health,
    generate_finetuned_response,
)
from app.finetuned_rag import generate_course_finetuned_rag_answer
from app.ollama import generate_base_model_response
from app.schemas import (
    BaseModelGenerateRequest,
    BaseModelGenerateResponse,
    FineTunedGenerateRequest,
    FineTunedGenerateResponse,
    FineTunedHealthResponse,
    FineTunedRagGenerateRequest,
    FineTunedRagGenerateResponse,
    CourseChunkMetadata,
    CourseChunksResponse,
    CourseSeedListResponse,
    FactAllocationCappedFact,
    FactAllocationItem,
    FactAllocationRankingItem,
    FactAllocationRequest,
    FactAllocationResponse,
    FactAllocationSkippedFact,
    FactAllocationSummary,
    FactInventoryItem,
    FactInventoryRequest,
    FactInventoryResponse,
    GeneratedSeedExample,
    ApprovedExportStatusResponse,
    PrepareTrainingSplitRequest,
    PrepareTrainingSplitResponse,
    RagGenerateRequest,
    RagGenerateResponse,
    RagGenerateSource,
    RagRetrieveResult,
    SeedExportApprovedRequest,
    SeedExportApprovedResponse,
    SeedGenerateRequest,
    SeedGenerateResponse,
    SeedQualityCheckRequest,
    SeedQualityCheckResponse,
    SeedReviewRecord,
    SeedReviewRequest,
    SeedReviewResponse,
    StarterSeedGenerateRequest,
    StarterSeedGenerateResponse,
    StarterSeedPersistence,
    StarterSeedProgress,
    StarterSeedTopUpRequest,
    StarterGenerationStatusResponse,
    SyllabusTextResponse,
    SyllabusUploadResponse,
)
from app.firebase_seeds import (
    FirebaseConfigurationError,
    course_seed_example_path,
    course_seed_examples_path,
    fetch_course_seed_examples,
    patch_course_seed_example,
)
from app.seed_dataset_quality import inspect_seed_dataset
from app.seed_export import FinetuneJsonlValidationError, export_approved_seeds
from app.seed_split import (
    DEFAULT_SPLIT_SEED,
    TrainingSplitError,
    approved_export_status,
    prepare_training_split,
)
from app.seed_review import REVIEW_STATUSES, apply_seed_review, resolve_review_status
from app.seed_allocation import allocate_slots
from app.seed_generation import generate_seeds_from_chunk, generate_starter_seeds_for_course
from app.fact_inventory_cache import load_or_build_fact_inventory
from app.ollama_coordination import (
    get_starter_job_status,
    starter_job_slot,
)
from app.starter_jobs import (
    run_auto_starter_seed_generation,
    try_queue_starter_seed_generation,
)
from app.storage import get_course_artifact_storage
from app.syllabus_extract import extract_clean_syllabus_text
from app.syllabus_upload import SyllabusUploadError, validate_syllabus_upload

app = FastAPI(title="Syllabus Model Lab Backend")

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "syllabus-model-lab-backend",
    }


@app.post("/base-model/generate", response_model=BaseModelGenerateResponse)
async def generate_base_model(
    request: BaseModelGenerateRequest,
) -> BaseModelGenerateResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        safe_course_id = assert_valid_course_id(request.course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # courseId is validated for route isolation but unused by the base model itself.
    result = await generate_base_model_response(question)
    return BaseModelGenerateResponse(
        answer=result["answer"],
        model=result["model"],
        responseType=result["response_type"],
        courseId=safe_course_id,
    )


@app.get("/fine-tuned/health", response_model=FineTunedHealthResponse)
async def fine_tuned_health() -> FineTunedHealthResponse:
    result = await check_finetuned_service_health()
    return FineTunedHealthResponse(
        status=result["status"],
        model=result.get("model"),
        adapterLoaded=result.get("adapterLoaded"),
        hostname=result.get("hostname"),
        port=result.get("port"),
        serviceUrl=result.get("serviceUrl"),
    )


@app.post("/fine-tuned/generate", response_model=FineTunedGenerateResponse)
async def generate_fine_tuned(
    request: FineTunedGenerateRequest,
) -> FineTunedGenerateResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        safe_course_id = assert_valid_course_id(request.course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # courseId is validated for route isolation; the remote service is course-agnostic.
    result = await generate_finetuned_response(question)
    return FineTunedGenerateResponse(
        answer=result["answer"],
        model=result["model"],
        responseType=result["response_type"],
        courseId=safe_course_id,
        adapterLoaded=result["adapter_loaded"],
        generationSeconds=result["generation_seconds"],
    )


@app.post("/rag/generate", response_model=RagGenerateResponse)
async def generate_rag_response(request: RagGenerateRequest) -> RagGenerateResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    result = await generate_course_rag_answer(
        course_id=request.course_id,
        question=question,
        top_k=request.top_k,
    )

    return RagGenerateResponse(
        courseId=result["courseId"],
        answer=result["answer"],
        model=result["model"],
        sources=[RagGenerateSource(**source) for source in result["sources"]],
        retrievedChunks=[
            RagRetrieveResult(**chunk) for chunk in result["retrievedChunks"]
        ],
        responseType=result["responseType"],
    )


@app.post("/fine-tuned-rag/generate", response_model=FineTunedRagGenerateResponse)
async def generate_fine_tuned_rag(
    request: FineTunedRagGenerateRequest,
) -> FineTunedRagGenerateResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    result = await generate_course_finetuned_rag_answer(
        course_id=request.course_id,
        question=question,
        top_k=request.top_k,
    )

    return FineTunedRagGenerateResponse(
        courseId=result["courseId"],
        answer=result["answer"],
        model=result["model"],
        sources=[RagGenerateSource(**source) for source in result["sources"]],
        retrievedChunks=[
            RagRetrieveResult(**chunk) for chunk in result["retrievedChunks"]
        ],
        responseType=result["responseType"],
        adapterLoaded=result["adapterLoaded"],
        generationSeconds=result["generationSeconds"],
    )


@app.post(
    "/api/courses/{course_id}/syllabus",
    response_model=SyllabusUploadResponse,
    status_code=201,
)
async def upload_course_syllabus(
    course_id: str,
    background_tasks: BackgroundTasks,
    syllabus_file: UploadFile = File(...),
) -> SyllabusUploadResponse:
    storage = get_course_artifact_storage()

    try:
        validated = await validate_syllabus_upload(course_id, syllabus_file)
    except SyllabusUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    finally:
        await syllabus_file.close()

    try:
        storage.save_original_syllabus(
            validated.course_id,
            validated.syllabus_type,
            validated.content,
        )
        extracted_text = extract_clean_syllabus_text(validated)
        storage.save_extracted_text(validated.course_id, extracted_text)
        index_data = await build_course_rag_index(
            course_id=validated.course_id,
            source_file=validated.original_filename,
            syllabus_text=extracted_text,
            storage=storage,
        )
    except SyllabusUploadError as exc:
        try:
            storage.delete_partial_files(validated.course_id)
        except Exception:
            pass
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:  # noqa: BLE001 - map unexpected storage failures to HTTP 500
        try:
            storage.delete_partial_files(validated.course_id)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="Could not save, extract, or index the syllabus file.",
        ) from exc

    queue_result = await try_queue_starter_seed_generation(validated.course_id)
    if queue_result.get("queued"):
        background_tasks.add_task(
            run_auto_starter_seed_generation,
            validated.course_id,
        )

    return SyllabusUploadResponse(
        courseId=validated.course_id,
        syllabusFileName=validated.original_filename,
        syllabusType=validated.syllabus_type,
        syllabusStatus="indexed",
        fileSize=validated.file_size,
        characterCount=len(extracted_text),
        chunkCount=int(index_data["chunkCount"]),
        starterSeedGenerationStatus=queue_result.get("status"),
    )


@app.get(
    "/api/courses/{course_id}/syllabus/text",
    response_model=SyllabusTextResponse,
)
def get_course_syllabus_text(course_id: str) -> SyllabusTextResponse:
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    storage = get_course_artifact_storage()
    text = storage.load_extracted_text(safe_course_id)
    if text is None:
        raise HTTPException(
            status_code=404,
            detail="Extracted syllabus text was not found for this course.",
        )

    return SyllabusTextResponse(
        courseId=safe_course_id,
        text=text,
        characterCount=len(text),
    )


@app.get(
    "/api/courses/{course_id}/chunks",
    response_model=CourseChunksResponse,
)
def get_course_chunks(course_id: str) -> CourseChunksResponse:
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    storage = get_course_artifact_storage()
    index_data = storage.load_index(safe_course_id)
    if index_data is None:
        raise HTTPException(
            status_code=404,
            detail="Course syllabus index was not found.",
        )

    chunks = [
        CourseChunkMetadata(
            chunkId=chunk["chunkId"],
            sectionTitle=chunk["sectionTitle"],
            text=chunk["text"],
            order=chunk["order"],
            documentTitle=chunk.get("documentTitle"),
            headingPath=chunk.get("headingPath"),
            startOffset=chunk.get("startOffset"),
            endOffset=chunk.get("endOffset"),
        )
        for chunk in index_data.get("chunks", [])
    ]

    return CourseChunksResponse(
        courseId=safe_course_id,
        chunkCount=len(chunks),
        chunks=chunks,
        indexVersion=index_data.get("indexVersion"),
        documentTitle=index_data.get("documentTitle"),
    )


@app.post(
    "/api/courses/{course_id}/seeds/generate",
    response_model=SeedGenerateResponse,
)
async def generate_course_seeds(
    course_id: str,
    request: SeedGenerateRequest,
) -> SeedGenerateResponse:
    """Temporary endpoint for testing AI seed generation from one chunk.

    Does not persist seeds to Firebase or trigger course creation.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await generate_seeds_from_chunk(
        course_id=safe_course_id,
        chunk_id=request.chunk_id,
        count=request.count,
    )

    return SeedGenerateResponse(
        courseId=result["courseId"],
        chunkId=result["chunkId"],
        model=result["model"],
        count=result["count"],
        seeds=[GeneratedSeedExample(**seed) for seed in result["seeds"]],
    )


@app.get(
    "/api/starter-generation/status",
    response_model=StarterGenerationStatusResponse,
)
async def starter_generation_status() -> StarterGenerationStatusResponse:
    """Read-only process-local status of the in-flight starter seed job."""
    status = get_starter_job_status()
    return StarterGenerationStatusResponse(
        active=bool(status["active"]),
        courseId=status["courseId"],
        operation=status["operation"],
        startedAt=status["startedAt"],
    )


@app.post(
    "/api/courses/{course_id}/seeds/generate-starter",
    response_model=StarterSeedGenerateResponse,
)
async def generate_course_starter_seeds(
    course_id: str,
    request: StarterSeedGenerateRequest,
) -> StarterSeedGenerateResponse:
    """Temporary endpoint for course-level starter seed generation.

    Set save=true to persist accepted validated seeds to Firebase.
    Does not trigger course creation.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    operation = "top_up" if request.top_up else "manual"
    async with starter_job_slot(safe_course_id, operation):
        result = await generate_starter_seeds_for_course(
            course_id=safe_course_id,
            target_count=request.target_count,
            save=request.save,
            force_refresh=request.force_refresh,
            top_up=request.top_up,
        )

    persistence = None
    if result.get("persistence") is not None:
        persistence = StarterSeedPersistence(**result["persistence"])

    return StarterSeedGenerateResponse(
        courseId=result["courseId"],
        model=result["model"],
        targetCount=result["targetCount"],
        seeds=[GeneratedSeedExample(**seed) for seed in result["seeds"]],
        progress=StarterSeedProgress(**result["progress"]),
        persistence=persistence,
        localSnapshotPath=result.get("localSnapshotPath"),
    )


@app.post(
    "/api/courses/{course_id}/seeds/top-up",
    response_model=StarterSeedGenerateResponse,
)
async def top_up_course_starter_seeds(
    course_id: str,
    request: StarterSeedTopUpRequest | None = None,
) -> StarterSeedGenerateResponse:
    """Fill the gap to targetCount without regenerating existing Firebase seeds.

    Reads courses/{courseId}/seedExamples, computes missingCount, generates only
    that many new accepted seeds, dedupes against existing questions, and saves
    only the new ones when save=true (default).
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    body = request or StarterSeedTopUpRequest()
    async with starter_job_slot(safe_course_id, "top_up"):
        result = await generate_starter_seeds_for_course(
            course_id=safe_course_id,
            target_count=body.target_count,
            save=body.save,
            force_refresh=body.force_refresh,
            top_up=True,
        )

    persistence = None
    if result.get("persistence") is not None:
        persistence = StarterSeedPersistence(**result["persistence"])

    return StarterSeedGenerateResponse(
        courseId=result["courseId"],
        model=result["model"],
        targetCount=result["targetCount"],
        seeds=[GeneratedSeedExample(**seed) for seed in result["seeds"]],
        progress=StarterSeedProgress(**result["progress"]),
        persistence=persistence,
        localSnapshotPath=result.get("localSnapshotPath"),
    )


@app.post(
    "/api/courses/{course_id}/facts/inventory",
    response_model=FactInventoryResponse,
)
async def get_course_fact_inventory(
    course_id: str,
    body: FactInventoryRequest | None = None,
) -> FactInventoryResponse:
    """Build or reuse an inspectable global fact inventory for a course syllabus.

    Extraction-only: this DOES NOT generate starter seeds. Shares the same
    per-course cache as starter generation. Pass forceRefresh to rebuild.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request = body or FactInventoryRequest()
    storage = get_course_artifact_storage()
    index_data = storage.load_index(safe_course_id)
    if index_data is None:
        raise HTTPException(
            status_code=404,
            detail="Course syllabus index was not found.",
        )

    raw_chunks = index_data.get("chunks", [])
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise HTTPException(
            status_code=404,
            detail=f'No syllabus chunks found for course "{safe_course_id}".',
        )

    inventory = await load_or_build_fact_inventory(
        course_id=safe_course_id,
        raw_chunks=raw_chunks,
        storage=storage,
        force_refresh=request.force_refresh,
    )

    return FactInventoryResponse(
        courseId=safe_course_id,
        model=inventory["model"],
        factCount=inventory["factCount"],
        droppedCount=inventory["droppedCount"],
        duplicatesRemoved=inventory.get("duplicatesRemoved", 0),
        fallbackUsed=inventory["fallbackUsed"],
        cached=bool(inventory.get("cached")),
        countsByScope=inventory["countsByScope"],
        countsByKind=inventory["countsByKind"],
        countsBySeries=inventory.get("countsBySeries", {}),
        facts=[FactInventoryItem(**fact) for fact in inventory["facts"]],
    )


@app.post(
    "/api/courses/{course_id}/facts/allocation",
    response_model=FactAllocationResponse,
)
async def get_course_fact_allocation(
    course_id: str,
    body: FactAllocationRequest | None = None,
) -> FactAllocationResponse:
    """Build/reuse fact inventory and allocate question slots (inspection).

    Allocation-only: this DOES NOT generate starter seeds, does not persist
    seeds, and is not wired into the live starter-generation pipeline.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request = body or FactAllocationRequest()
    target_count = request.target_count

    storage = get_course_artifact_storage()
    index_data = storage.load_index(safe_course_id)
    if index_data is None:
        raise HTTPException(
            status_code=404,
            detail="Course syllabus index was not found.",
        )

    raw_chunks = index_data.get("chunks", [])
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise HTTPException(
            status_code=404,
            detail=f'No syllabus chunks found for course "{safe_course_id}".',
        )

    inventory = await load_or_build_fact_inventory(
        course_id=safe_course_id,
        raw_chunks=raw_chunks,
        storage=storage,
        force_refresh=request.force_refresh,
    )
    allocation = allocate_slots(
        inventory["facts"],
        target_count=target_count,
    )
    summary = allocation["summary"]

    return FactAllocationResponse(
        courseId=safe_course_id,
        model=inventory["model"],
        factCount=inventory["factCount"],
        droppedCount=inventory["droppedCount"],
        duplicatesRemoved=inventory.get("duplicatesRemoved", 0),
        fallbackUsed=inventory["fallbackUsed"],
        facts=[FactInventoryItem(**fact) for fact in inventory["facts"]],
        allocations=[
            FactAllocationItem(**item) for item in allocation["allocations"]
        ],
        summary=FactAllocationSummary(
            targetCount=summary["targetCount"],
            allocatedSlots=summary["allocatedSlots"],
            byScope=summary["byScope"],
            byKind=summary["byKind"],
            bySeries=summary["bySeries"],
            skippedFacts=[
                FactAllocationSkippedFact(**item)
                for item in summary["skippedFacts"]
            ],
            cappedFacts=[
                FactAllocationCappedFact(**item) for item in summary["cappedFacts"]
            ],
            caps=summary["caps"],
            courseWideAllocated=summary["courseWideAllocated"],
            courseWideReserve=summary["courseWideReserve"],
        ),
        ranking=[
            FactAllocationRankingItem(**item) for item in allocation["ranking"]
        ],
    )


def _seed_records_from_firebase_payload(payload: dict) -> list[dict]:
    records: list[dict] = []
    for seed_id, raw in payload.items():
        if not isinstance(raw, dict):
            continue
        record = dict(raw)
        record.setdefault("id", seed_id)
        records.append(record)
    return records


@app.get(
    "/api/courses/{course_id}/seeds",
    response_model=CourseSeedListResponse,
)
async def list_course_seeds(course_id: str) -> CourseSeedListResponse:
    """List Firebase seedExamples for review (course-scoped path only)."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        payload = await fetch_course_seed_examples(safe_course_id)
    except FirebaseConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    seeds = _seed_records_from_firebase_payload(payload)
    return CourseSeedListResponse(
        courseId=safe_course_id,
        count=len(seeds),
        firebasePath=course_seed_examples_path(safe_course_id),
        seeds=[SeedReviewRecord(**seed) for seed in seeds],
    )


@app.post(
    "/api/courses/{course_id}/seeds/{seed_id}/review",
    response_model=SeedReviewResponse,
)
async def review_course_seed(
    course_id: str,
    seed_id: str,
    body: SeedReviewRequest,
) -> SeedReviewResponse:
    """Approve, reject, or edit one seed. Edits preserve grounding provenance."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    status = body.review_status.strip().lower()
    if status not in REVIEW_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"reviewStatus must be one of {sorted(REVIEW_STATUSES)}.",
        )

    try:
        payload = await fetch_course_seed_examples(safe_course_id)
    except FirebaseConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    existing = payload.get(seed_id)
    if not isinstance(existing, dict):
        raise HTTPException(status_code=404, detail=f'Seed "{seed_id}" was not found.')

    try:
        updated = apply_seed_review(
            {**existing, "id": seed_id},
            review_status=status,
            question=body.question,
            answer=body.answer,
            review_notes=body.review_notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    stored = await patch_course_seed_example(safe_course_id, seed_id, updated)
    return SeedReviewResponse(
        courseId=safe_course_id,
        seedId=seed_id,
        seed=SeedReviewRecord(**stored),
        firebasePath=course_seed_example_path(safe_course_id, seed_id),
    )


@app.post(
    "/api/courses/{course_id}/seeds/quality-check",
    response_model=SeedQualityCheckResponse,
)
async def quality_check_course_seeds(
    course_id: str,
    body: SeedQualityCheckRequest | None = None,
) -> SeedQualityCheckResponse:
    """Inspect course seedExamples for coverage and quality flags."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request = body or SeedQualityCheckRequest()
    try:
        payload = await fetch_course_seed_examples(safe_course_id)
    except FirebaseConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    seeds = _seed_records_from_firebase_payload(payload)
    if request.review_statuses:
        allowed = {
            item.strip().lower()
            for item in request.review_statuses
            if isinstance(item, str) and item.strip()
        }
        seeds = [
            seed for seed in seeds if resolve_review_status(seed) in allowed
        ]

    report = inspect_seed_dataset(seeds)
    return SeedQualityCheckResponse(
        courseId=safe_course_id,
        firebasePath=course_seed_examples_path(safe_course_id),
        report=report,
    )


@app.post(
    "/api/courses/{course_id}/seeds/export-approved",
    response_model=SeedExportApprovedResponse,
)
async def export_approved_course_seeds(
    course_id: str,
    body: SeedExportApprovedRequest | None = None,
) -> SeedExportApprovedResponse:
    """Export approved-only JSONL + metadata under data/exports/{courseId}/."""
    del body  # reserved; approved-only is always enforced
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        payload = await fetch_course_seed_examples(safe_course_id)
    except FirebaseConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    seeds = _seed_records_from_firebase_payload(payload)
    try:
        summary = export_approved_seeds(course_id=safe_course_id, seeds=seeds)
    except FinetuneJsonlValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SeedExportApprovedResponse(courseId=safe_course_id, summary=summary)


@app.get(
    "/api/courses/{course_id}/seeds/approved-export-status",
    response_model=ApprovedExportStatusResponse,
)
async def get_approved_export_status(course_id: str) -> ApprovedExportStatusResponse:
    """Report whether approved-finetune.jsonl exists for this course."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    status = approved_export_status(safe_course_id)
    return ApprovedExportStatusResponse(
        courseId=status["courseId"],
        exists=status["exists"],
        exportPath=status["exportPath"],
        exampleCount=status["exampleCount"],
        sourceFile=status["sourceFile"],
    )


@app.post(
    "/api/courses/{course_id}/seeds/prepare-training-split",
    response_model=PrepareTrainingSplitResponse,
)
async def prepare_course_training_split(
    course_id: str,
    body: PrepareTrainingSplitRequest | None = None,
) -> PrepareTrainingSplitResponse:
    """Create deterministic train/validation split from approved-finetune.jsonl."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    split_seed = DEFAULT_SPLIT_SEED
    if body is not None and body.split_seed is not None:
        split_seed = int(body.split_seed)

    try:
        summary = prepare_training_split(
            safe_course_id,
            split_seed=split_seed,
        )
    except TrainingSplitError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PrepareTrainingSplitResponse(courseId=safe_course_id, summary=summary)
